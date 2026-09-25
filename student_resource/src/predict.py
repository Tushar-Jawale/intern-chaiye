"""
Phase 5 — Score blocking candidates and write matching_results.tsv.

Every Source-1 row in the candidate file is written, including singletons.
A match is kept only when its score is at least the trained F0.5 threshold,
so the matched ids are always a subset of that row's candidates.

Candidates are scored in batches (one booster call per ~200k pairs). On POSIX
the file is split into contiguous spans, one per worker; each worker writes its
own part file and reports progress, and the parts are concatenated in order.
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


def _span_rows(path: str, start: int, end: int):
    for i, row in enumerate(iter_candidates(path)):
        if i < start:
            continue
        if i >= end:
            break
        yield row


_PRED: dict = {}


def _predict_span(job: tuple[int, int, int]) -> tuple[int, str, int, int]:
    import lightgbm as lgb

    idx, start, end = job
    state = _PRED
    booster = lgb.Booster(model_file=state["model"])
    part = f"{state['output']}.part{idx}"
    written = matched = 0
    with open(part, "w", encoding="utf-8", newline="\n") as handle:
        rows = _span_rows(state["candidates"], start, end)
        for eid, chosen in score_stream(booster, state["records"], rows, state["threshold"], f"worker {idx}", end - start):
            handle.write(eid + "\t" + ",".join(chosen) + "\n")
            written += 1
            matched += len(chosen)
    return idx, part, written, matched


def run(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    with open(args.meta, encoding="utf-8") as handle:
        meta = json.load(handle)
    threshold = float(meta["threshold"])
    print(f"Threshold {threshold:.3f}", flush=True)
    names = ("test_s1", "test_s2", "test_s3") if args.split == "test" else ("train_s1", "train_s2", "train_s3")
    print("Collecting ids", flush=True)
    need, n_rows = collect_need(args.candidates)
    print(f"Loading {len(need):,} records", flush=True)
    records = load_records(args.preprocessed, need, names=names)
    del need
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    workers = getattr(args, "workers", 0) or (4 if os.name == "posix" and n_rows >= 4000 else 1)
    print(f"Scoring {n_rows:,} entities on {workers} core(s)", flush=True)
    t0 = time.time()
    matched = written = 0
    if workers == 1:
        booster = lgb.Booster(model_file=args.model)
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("source1_entity_id\tmatched_entity_ids\n")
            for eid, chosen in score_stream(booster, records, iter_candidates(args.candidates), threshold, "main", n_rows):
                handle.write(eid + "\t" + ",".join(chosen) + "\n")
                written += 1
                matched += len(chosen)
    else:
        import multiprocessing as mp

        step = (n_rows + workers - 1) // workers
        jobs = [(k, s, min(s + step, n_rows)) for k, s in enumerate(range(0, n_rows, step))]
        _PRED.clear()
        _PRED.update(candidates=args.candidates, model=args.model, records=records, threshold=threshold, output=args.output)
        ctx = mp.get_context("fork")
        with ctx.Pool(len(jobs)) as pool:
            results = sorted(pool.map(_predict_span, jobs))
        _PRED.clear()
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("source1_entity_id\tmatched_entity_ids\n")
            for _, part, n_written, n_matched in results:
                with open(part, encoding="utf-8") as src:
                    for line in src:
                        handle.write(line)
                os.remove(part)
                written += n_written
                matched += n_matched
    print(f"Wrote {args.output}  entities={written:,}  matches={matched:,}  in {(time.time() - t0) / 60:.1f} min", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write matching_results.tsv")
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--workers", type=int, default=0, help="0 = 4 on Linux, 1 elsewhere")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
