"""
Phase 5 — Score blocking candidates and write matching_results.tsv.

Every Source-1 row in the candidate file is written, including singletons.
A match is kept only when its score is at least the trained F0.5 threshold,
so the matched ids are always a subset of that row's candidates.

Candidates are scored in batches (one booster call per ~200k pairs).

Memory: the candidate file is split into job files grouped by the Source-1
country, and each job loads only the records its own rows reference. No
process ever holds all ~11M records, and nothing large is shared across a
fork (sharing a big dict across forked workers duplicates it page by page and
got the workers OOM-killed, which leaves a multiprocessing pool hanging).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

try:
    from features import pair_features
    from train import load_records
except ImportError:
    pass

BATCH_PAIRS = 200_000
REPORT_EVERY = 50_000


def iter_candidates(path: str):
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()
        if "source1_entity_id" not in header:
            raise ValueError(f"Unexpected header: {header!r}")
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            eid, rest = line.split("\t", 1)
            cands = [c for c in rest.split(",") if c] if rest else []
            yield eid, cands


def collect_need(path: str) -> tuple[set[str], int]:
    need = set()
    n = 0
    for eid, cands in iter_candidates(path):
        need.add(eid)
        need.update(cands)
        n += 1
    return need, n


def unique(ids: list[str]) -> list[str]:
    seen = set()
    out = []
    for eid in ids:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def score_stream(booster, records: dict, rows, threshold: float, label: str, total: int):
    """Yield (eid, matched ids) for each (eid, candidates) in order, batching model calls."""
    empty = ("", "", "")
    pending: list[tuple[str, list[str]]] = []
    feats: list[list[float]] = []
    t0 = time.time()
    done = 0

    def flush():
        scores = booster.predict(np.asarray(feats, dtype=np.float32)) if feats else np.zeros(0)
        pos = 0
        for eid, keep in pending:
            k = len(keep)
            chosen = [cid for cid, s in zip(keep, scores[pos:pos + k]) if s >= threshold]
            pos += k
            yield eid, unique(chosen)
        pending.clear()
        feats.clear()

    for eid, cands in rows:
        left = records.get(eid, empty)
        keep = []
        for cid in cands:
            right = records.get(cid)
            if right is None:
                continue
            feats.append(pair_features(left[0], left[1], left[2], right[0], right[1], right[2]))
            keep.append(cid)
        pending.append((eid, keep))
        if len(feats) >= BATCH_PAIRS:
            for item in flush():
                done += 1
                if done % REPORT_EVERY == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    print(f"  [{label}] {done:,}/{total:,} entities  {rate:.0f}/s  eta {eta / 60:.0f} min", flush=True)
                yield item
    for item in flush():
        done += 1
        yield item
    print(f"  [{label}] {done:,}/{total:,} entities done in {(time.time() - t0) / 60:.1f} min", flush=True)


def load_s1_countries(preprocessed: str, split: str) -> dict[str, str]:
    import pyarrow.parquet as pq

    path = os.path.join(preprocessed, f"{split}_s1.parquet")
    out: dict[str, str] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=500_000, columns=["entity_id", "country"]):
        data = batch.to_pydict()
        for eid, country in zip(data["entity_id"], data["country"]):
            out[eid] = "" if country is None else str(country)
    return out


def split_jobs(candidates: str, countries: dict[str, str], workers: int, work_dir: str) -> tuple[list[dict], int]:
    """Write one input file per job. Rows of one country stay together so a
    job only needs that country's records."""
    per_country: dict[str, int] = {}
    n_rows = 0
    for eid, _ in iter_candidates(candidates):
        c = countries.get(eid, "")
        per_country[c] = per_country.get(c, 0) + 1
        n_rows += 1
    target = max(1, -(-n_rows // max(workers, 1)))
    chunk_size: dict[str, int] = {}
    for c, n in per_country.items():
        k = max(1, -(-n // target))
        chunk_size[c] = -(-n // k)
    os.makedirs(work_dir, exist_ok=True)
    handles: dict[tuple[str, int], object] = {}
    jobs: dict[tuple[str, int], dict] = {}
    seen: dict[str, int] = {}
    with open(candidates, encoding="utf-8") as src:
        src.readline()
        for line in src:
            if not line.strip():
                continue
            eid = line.split("\t", 1)[0]
            c = countries.get(eid, "")
            i = seen.get(c, 0)
            seen[c] = i + 1
            key = (c, i // chunk_size[c])
            h = handles.get(key)
            if h is None:
                idx = len(jobs)
                path = os.path.join(work_dir, f"job{idx}.in.tsv")
                h = open(path, "w", encoding="utf-8", newline="\n")
                h.write("source1_entity_id\tcandidate_entity_ids\n")
                handles[key] = h
                jobs[key] = {"idx": idx, "label": f"{c or '?'}#{key[1]}", "input": path, "rows": 0}
            h.write(line if line.endswith("\n") else line + "\n")
            jobs[key]["rows"] += 1
    for h in handles.values():
        h.close()
    return sorted(jobs.values(), key=lambda j: -j["rows"]), n_rows


_PRED: dict = {}


def _predict_job(job: dict) -> tuple[int, str, int, int]:
    import lightgbm as lgb

    state = _PRED
    t0 = time.time()
    need, n_rows = collect_need(job["input"])
    records = load_records(state["preprocessed"], need, names=state["names"])
    del need
    print(f"  [{job['label']}] loaded {len(records):,} records for {n_rows:,} entities in {time.time() - t0:.0f}s", flush=True)
    booster = lgb.Booster(model_file=state["model"])
    part = job["input"].replace(".in.tsv", ".out.tsv")
    written = matched = 0
    with open(part, "w", encoding="utf-8", newline="\n") as handle:
        rows = iter_candidates(job["input"])
        for eid, chosen in score_stream(booster, records, rows, state["threshold"], job["label"], n_rows):
            handle.write(eid + "\t" + ",".join(chosen) + "\n")
            written += 1
            matched += len(chosen)
    del records
    os.remove(job["input"])
    return job["idx"], part, written, matched


def run(args: argparse.Namespace) -> None:
    with open(args.meta, encoding="utf-8") as handle:
        meta = json.load(handle)
    threshold = float(meta["threshold"])
    print(f"Threshold {threshold:.3f}", flush=True)
    split = args.split
    names = (f"{split}_s1", f"{split}_s2", f"{split}_s3")
    workers = getattr(args, "workers", 0) or (4 if os.name == "posix" else 1)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    work_dir = os.path.join(out_dir, "_predict_jobs")
    os.makedirs(out_dir, exist_ok=True)

    print("Splitting candidates by country", flush=True)
    countries = load_s1_countries(args.preprocessed, split)
    jobs, n_rows = split_jobs(args.candidates, countries, workers, work_dir)
    del countries
    for job in jobs:
        print(f"  job {job['label']}: {job['rows']:,} entities", flush=True)
    print(f"Scoring {n_rows:,} entities in {len(jobs)} job(s) on {workers} process(es)", flush=True)

    _PRED.clear()
    _PRED.update(preprocessed=args.preprocessed, names=names, model=args.model, threshold=threshold)
    t0 = time.time()
    if workers == 1 or len(jobs) == 1:
        results = [_predict_job(job) for job in jobs]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(min(workers, len(jobs)), maxtasksperchild=1) as pool:
            results = pool.map(_predict_job, jobs, chunksize=1)
    _PRED.clear()

    written = matched = 0
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for _, part, n_written, n_matched in sorted(results):
            with open(part, encoding="utf-8") as src:
                for line in src:
                    handle.write(line)
            os.remove(part)
            written += n_written
            matched += n_matched
    try:
        os.rmdir(work_dir)
    except OSError:
        pass
    if written != n_rows:
        raise RuntimeError(f"wrote {written:,} rows but the candidate file has {n_rows:,}")
    print(f"Wrote {args.output}  entities={written:,}  matches={matched:,}  in {(time.time() - t0) / 60:.1f} min", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write matching_results.tsv")
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--workers", type=int, default=0, help="0 = 4 on Linux, 1 elsewhere; lower it if memory is tight")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
