"""
Phase 5 — Score blocking candidates and write matching_results.tsv.

Every Source-1 row in the candidate file is written, including singletons.
A match is kept only when its score is at least the trained F0.5 threshold,
so the matched ids are always a subset of that row's candidates.

Speed: record tokens are prepared once per record (features.prep_record), the
two fuzzy ratios are computed in C++ per batch (rapidfuzz cpdist), and the
booster is called once per ~200k pairs with LightGBM prediction early stopping.
Each job scores its first batch with and without early stopping and prints how
many match decisions differ.

--top-k K keeps only the first K candidates of each row (rows are best-first)
and rewrites candidate_pairs.tsv to exactly that list, so the candidate file is
still the model input; the original is kept as candidate_pairs_full.tsv.

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
    from features import batch_features, prep_record
    from train import load_records
except ImportError:
    pass

BATCH_PAIRS = 200_000
REPORT_EVERY = 50_000
EARLY_STOP = {"pred_early_stop": True, "pred_early_stop_freq": 10, "pred_early_stop_margin": 10.0}


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


def score_stream(booster, records: dict, rows, threshold: float, label: str, total: int, num_threads: int = 0):
    """Yield (eid, matched ids) for each (eid, candidates) in order, batching model calls.
    ``records`` maps id -> features.prep_record(...) tuple."""
    empty = prep_record("", "", "")
    pending: list[tuple[str, list[str]]] = []
    lefts: list[tuple] = []
    rights: list[tuple] = []
    t0 = time.time()
    state = {"done": 0, "next_report": REPORT_EVERY, "audited": False}

    def flush():
        x = batch_features(lefts, rights, workers=max(num_threads, 1))
        if len(x):
            scores = booster.predict(x, num_threads=num_threads, **EARLY_STOP)
            if not state["audited"]:
                full = booster.predict(x, num_threads=num_threads)
                flips = int(((full >= threshold) != (scores >= threshold)).sum())
                print(f"  [{label}] early-stop audit: {flips} of {len(x):,} match decisions differ from full prediction", flush=True)
                state["audited"] = True
        else:
            scores = np.zeros(0)
        pos = 0
        out = []
        for eid, keep in pending:
            k = len(keep)
            out.append((eid, unique([cid for cid, s in zip(keep, scores[pos:pos + k]) if s >= threshold])))
            pos += k
        pending.clear()
        lefts.clear()
        rights.clear()
        state["done"] += len(out)
        if state["done"] >= state["next_report"] and state["done"] < total:
            rate = state["done"] / max(time.time() - t0, 1e-9)
            eta = (total - state["done"]) / max(rate, 1e-9)
            print(f"  [{label}] {state['done']:,}/{total:,} entities  {rate:.0f}/s  eta {eta / 60:.0f} min", flush=True)
            while state["next_report"] <= state["done"]:
                state["next_report"] += REPORT_EVERY
        return out

    for eid, cands in rows:
        left = records.get(eid, empty)
        keep = []
        for cid in cands:
            right = records.get(cid)
            if right is None:
                continue
            lefts.append(left)
            rights.append(right)
            keep.append(cid)
        pending.append((eid, keep))
        if len(lefts) >= BATCH_PAIRS:
            yield from flush()
    yield from flush()
    print(f"  [{label}] {state['done']:,}/{total:,} entities done in {(time.time() - t0) / 60:.1f} min", flush=True)


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


def _truncate(line: str, top_k: int) -> str:
    line = line.rstrip("\n")
    if top_k <= 0:
        return line + "\n"
    eid, _, rest = line.partition("\t")
    cands = [c for c in rest.split(",") if c][:top_k]
    return eid + "\t" + ",".join(cands) + "\n"


def split_jobs(candidates: str, countries: dict[str, str], workers: int, work_dir: str,
               top_k: int = 0, truncated_out: str | None = None) -> tuple[list[dict], int]:
    """Write one input file per job. Rows of one country stay together so a
    job only needs that country's records. With top_k, rows are cut to their
    first top_k candidates and the cut file is also written to truncated_out."""
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
    cut = None
    if truncated_out:
        cut = open(truncated_out, "w", encoding="utf-8", newline="\n")
        cut.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(candidates, encoding="utf-8") as src:
        src.readline()
        for line in src:
            if not line.strip():
                continue
            line = _truncate(line, top_k)
            if cut is not None:
                cut.write(line)
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
            h.write(line)
            jobs[key]["rows"] += 1
    for h in handles.values():
        h.close()
    if cut is not None:
        cut.close()
    return sorted(jobs.values(), key=lambda j: -j["rows"]), n_rows


_PRED: dict = {}


def _predict_job(job: dict) -> tuple[int, str, int, int]:
    import lightgbm as lgb

    state = _PRED
    t0 = time.time()
    need, n_rows = collect_need(job["input"])
    records = load_records(state["preprocessed"], need, names=state["names"])
    del need
    for key in records:
        records[key] = prep_record(*records[key])
    print(f"  [{job['label']}] loaded {len(records):,} records for {n_rows:,} entities in {time.time() - t0:.0f}s", flush=True)
    booster = lgb.Booster(model_file=state["model"])
    part = job["input"].replace(".in.tsv", ".out.tsv")
    written = matched = 0
    with open(part, "w", encoding="utf-8", newline="\n") as handle:
        rows = iter_candidates(job["input"])
        for eid, chosen in score_stream(booster, records, rows, state["threshold"], job["label"], n_rows, state["num_threads"]):
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

    top_k = getattr(args, "top_k", 0) or 0
    truncated_tmp = None
    if top_k > 0:
        truncated_tmp = args.candidates + ".tmp"
        print(f"Keeping the first {top_k} candidates of each row", flush=True)

    print("Splitting candidates by country", flush=True)
    countries = load_s1_countries(args.preprocessed, split)
    jobs, n_rows = split_jobs(args.candidates, countries, workers, work_dir, top_k, truncated_tmp)
    del countries
    if truncated_tmp:
        full_path = args.candidates[:-4] + "_full.tsv" if args.candidates.endswith(".tsv") else args.candidates + ".full"
        os.replace(args.candidates, full_path)
        os.replace(truncated_tmp, args.candidates)
        print(f"  {args.candidates} now holds the top-{top_k} candidates the model scores; original kept at {full_path}", flush=True)
    for job in jobs:
        print(f"  job {job['label']}: {job['rows']:,} entities", flush=True)
    print(f"Scoring {n_rows:,} entities in {len(jobs)} job(s) on {workers} process(es)", flush=True)

    _PRED.clear()
    _PRED.update(preprocessed=args.preprocessed, names=names, model=args.model, threshold=threshold,
                 num_threads=1 if workers > 1 else 0)
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
    p.add_argument("--top-k", type=int, default=0, help="Score only the first K candidates per row and rewrite the candidate file to match (0 = all)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
