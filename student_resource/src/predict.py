"""
Phase 5 — Score blocking candidates and write matching_results.tsv.

Every Source-1 row in the candidate file is written, including singletons.
A match is kept only when its score is at least the trained F0.5 threshold,
so the matched ids are always a subset of that row's candidates.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    from features import pair_features
    from train import load_records
except ImportError:
    pass


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


_PRED: dict = {}


def _predict_span(bounds: tuple[int, int]) -> list[tuple[str, list[str]]]:
    import lightgbm as lgb

    start, end = bounds
    state = _PRED
    booster = lgb.Booster(model_file=state["model"])
    records = state["records"]
    threshold = state["threshold"]
    rows = []
    with open(state["candidates"], encoding="utf-8") as handle:
        handle.readline()
        for i, line in enumerate(handle):
            if i < start:
                continue
            if i >= end:
                break
            line = line.rstrip("\n")
            eid, rest = line.split("\t", 1)
            cands = [c for c in rest.split(",") if c] if rest else []
            chosen = score_candidates(booster, records, eid, cands, threshold) if cands else []
            rows.append((eid, chosen))
    return rows


def unique(ids: list[str]) -> list[str]:
    seen = set()
    out = []
    for eid in ids:
        if eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def score_candidates(booster, records: dict, eid: str, cands: list[str], threshold: float) -> list[str]:
    left = records.get(eid, ("", "", ""))
    rows = []
    keep = []
    for cid in cands:
        right = records.get(cid)
        if right is None:
            continue
        rows.append(pair_features(left[0], left[1], left[2], right[0], right[1], right[2]))
        keep.append(cid)
    if not rows:
        return []
    scores = booster.predict(np.asarray(rows, dtype=np.float32))
    return unique([cid for cid, score in zip(keep, scores) if float(score) >= threshold])


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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    workers = 4 if os.name == "posix" and n_rows >= 4000 else 1
    print(f"Scoring {n_rows:,} entities on {workers} core(s)", flush=True)
    matched = 0
    written = 0
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        if workers == 1:
            booster = lgb.Booster(model_file=args.model)
            for eid, cands in iter_candidates(args.candidates):
                chosen = score_candidates(booster, records, eid, cands, threshold) if cands else []
                handle.write(eid + "\t" + ",".join(chosen) + "\n")
                written += 1
                matched += len(chosen)
        else:
            import multiprocessing as mp

            step = (n_rows + workers - 1) // workers
            spans = [(i, min(i + step, n_rows)) for i in range(0, n_rows, step)]
            _PRED.clear()
            _PRED.update(
                candidates=args.candidates,
                model=args.model,
                records=records,
                threshold=threshold,
            )
            ctx = mp.get_context("fork")
            with ctx.Pool(len(spans)) as pool:
                for part in pool.imap(_predict_span, spans):
                    for eid, chosen in part:
                        handle.write(eid + "\t" + ",".join(chosen) + "\n")
                        written += 1
                        matched += len(chosen)
            _PRED.clear()
    print(f"Wrote {args.output}  entities={written:,}  matches={matched:,}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write matching_results.tsv")
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["test", "train"], default="test")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
