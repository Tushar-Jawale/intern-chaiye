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


def collect_need(path: str) -> set[str]:
    need = set()
    for eid, cands in iter_candidates(path):
        need.add(eid)
        need.update(cands)
    return need


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
    need = collect_need(args.candidates)
    print(f"Loading {len(need):,} records", flush=True)
    records = load_records(args.preprocessed, need, names=names)
    booster = lgb.Booster(model_file=args.model)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n = 0
    matched = 0
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for eid, cands in iter_candidates(args.candidates):
            chosen = score_candidates(booster, records, eid, cands, threshold) if cands else []
            handle.write(eid + "\t" + ",".join(chosen) + "\n")
            n += 1
            matched += len(chosen)
            if n % 50000 == 0:
                print(f"  scored {n:,} entities, matches so far {matched:,}", flush=True)
    print(f"Wrote {args.output}  entities={n:,}  matches={matched:,}", flush=True)


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
