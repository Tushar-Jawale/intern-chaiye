"""
End-to-end pipeline.

  block train sample  ->  train matcher  ->  block test  ->  write both submission files

Train blocking scores a 100,000-row Source-1 sample against ALL train
candidates (that is where blocking recall and the rank histogram are printed).
Test blocking covers every test Source-1 entity. Re-running skips a step when
its output file is already present; delete the file to redo the step.

    python src/run_basic.py --preprocessed preprocessed --output output

`--self-test` runs the same four steps on a tiny in-memory dataset.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from argparse import Namespace

import blocking
import predict
import train


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    train_candidates = os.path.join(args.output, "sample_candidate_pairs.tsv")
    test_candidates = os.path.join(args.output, "candidate_pairs.tsv")
    model_path = os.path.join(args.output, "matcher.txt")
    meta_path = os.path.join(args.output, "matcher_meta.json")
    matching_path = os.path.join(args.output, "matching_results.tsv")

    if args.model:
        model_path = args.model
        meta_path = args.meta
        print(f"BLOCK TRAIN skipped. Using {model_path}", flush=True)
    elif os.path.exists(model_path):
        print(f"BLOCK TRAIN skipped. Using {model_path}", flush=True)
    elif not os.path.exists(train_candidates):
        print("BLOCK TRAIN (sample)", flush=True)
        blocking.run(Namespace(
            mode="sample",
            sample_size=args.sample_size,
            top_k=args.top_k,
            min_score=args.min_score,
            country="",
            preprocessed=args.preprocessed,
            output=args.output,
        ))
    else:
        print(f"BLOCK TRAIN skipped, using {train_candidates}", flush=True)

    if args.model or os.path.exists(model_path):
        if not args.model:
            print(f"TRAIN skipped, using {model_path}", flush=True)
    elif not os.path.exists(model_path):
        print("TRAIN", flush=True)
        train.run(Namespace(
            preprocessed=args.preprocessed,
            candidates=train_candidates,
            output=args.output,
            max_entities=args.max_entities,
            neg_per_pos=args.neg_per_pos,
            class_weight=False,
            n_estimators=args.n_estimators,
            seed=args.seed,
        ))
    else:
        print(f"TRAIN skipped, using {model_path}", flush=True)

    if not os.path.exists(test_candidates):
        print("BLOCK TEST", flush=True)
        blocking.run(Namespace(
            mode="test",
            sample_size=args.sample_size,
            top_k=args.top_k,
            min_score=args.min_score,
            country="",
            preprocessed=args.preprocessed,
            output=args.output,
        ))
    else:
        print(f"BLOCK TEST skipped, using {test_candidates}", flush=True)

    print("PREDICT", flush=True)
    predict.run(Namespace(
        preprocessed=args.preprocessed,
        candidates=test_candidates,
        model=model_path,
        meta=meta_path,
        output=matching_path,
        split="test",
    ))
    print(f"Submission files:\n  {matching_path}\n  {test_candidates}", flush=True)


def _frame(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def self_test() -> None:
    root = tempfile.mkdtemp(prefix="basic_er_")
    pre = os.path.join(root, "preprocessed")
    out = os.path.join(root, "output")
    os.makedirs(pre)
    s1 = [
        {"entity_id": "S1-1", "country": "US", "name_core": "alpha bakery", "addr_normalized": "10 oak street austin", "addr_numbers": "10"},
        {"entity_id": "S1-2", "country": "US", "name_core": "other cafe", "addr_normalized": "99 pine road dallas", "addr_numbers": "99"},
        {"entity_id": "S1-3", "country": "US", "name_core": "west market", "addr_normalized": "551 pine road dallas", "addr_numbers": "551"},
        {"entity_id": "S1-4", "country": "US", "name_core": "alpha bakery outlet", "addr_normalized": "10 oak street austin", "addr_numbers": "10"},
    ]
    s2 = [
        {"entity_id": "S2-1", "country": "US", "name_core": "alpha bakery", "addr_normalized": "10 oak street austin", "addr_numbers": "10"},
        {"entity_id": "S2-3", "country": "US", "name_core": "west market", "addr_normalized": "551 pine road dallas", "addr_numbers": "551"},
        {"entity_id": "S2-9", "country": "US", "name_core": "noise shop", "addr_normalized": "1 main street boston", "addr_numbers": "1"},
    ]
    s3 = [
        {"entity_id": "S3-4", "country": "US", "name_core": "alpha bakery outlet", "addr_normalized": "10 oak street austin", "addr_numbers": "10"},
    ]
    _frame(s1).to_parquet(os.path.join(pre, "train_s1.parquet"), index=False)
    _frame(s2).to_parquet(os.path.join(pre, "train_s2.parquet"), index=False)
    _frame(s3).to_parquet(os.path.join(pre, "train_s3.parquet"), index=False)
    _frame(s1).to_parquet(os.path.join(pre, "test_s1.parquet"), index=False)
    _frame(s2).to_parquet(os.path.join(pre, "test_s2.parquet"), index=False)
    _frame(s3).to_parquet(os.path.join(pre, "test_s3.parquet"), index=False)
    _frame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-3"},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S3-4"},
    ]).to_parquet(os.path.join(pre, "train_gt.parquet"), index=False)

    run(Namespace(
        preprocessed=pre,
        output=out,
        sample_size=10,
        top_k=20,
        min_score=0.0,
        max_entities=10,
        neg_per_pos=0,
        n_estimators=50,
        seed=42,
        model="",
        meta="",
    ))
    matching = os.path.join(out, "matching_results.tsv")
    candidates = os.path.join(out, "candidate_pairs.tsv")
    got = {}
    cands = {}
    with open(matching, encoding="utf-8") as handle:
        assert handle.readline().strip() == "source1_entity_id\tmatched_entity_ids"
        for line in handle:
            eid, rest = line.rstrip("\n").split("\t")
            got[eid] = [x for x in rest.split(",") if x]
    with open(candidates, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            eid, rest = line.rstrip("\n").split("\t")
            cands[eid] = set(x for x in rest.split(",") if x)
    assert set(got) == {"S1-1", "S1-2", "S1-3", "S1-4"}, got
    for eid, matches in got.items():
        assert set(matches) <= cands[eid], (eid, matches, cands[eid])
    print("basic-model self-test OK", got)
    shutil.rmtree(root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the basic entity-resolution model")
    p.add_argument("--preprocessed", default="")
    p.add_argument("--output", default="")
    p.add_argument("--model", default="", help="Reuse a trained matcher.txt (skips train blocking + training)")
    p.add_argument("--meta", default="", help="matcher_meta.json that goes with --model")
    p.add_argument("--sample-size", type=int, default=100000, help="Source-1 rows blocked for training")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--min-score", type=float, default=1.0)
    p.add_argument("--max-entities", type=int, default=100000, help="Entities used for fit/threshold/report")
    p.add_argument("--neg-per-pos", type=int, default=0, help="0 = every candidate is a negative")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    args.model = args.model.strip()
    args.meta = args.meta.strip()
    if args.model and not args.meta:
        raise SystemExit("Pass --meta with --model")
    return args


if __name__ == "__main__":
    cli = parse_args()
    if cli.self_test:
        self_test()
    else:
        if not cli.preprocessed or not cli.output:
            raise SystemExit("Pass --preprocessed and --output, or --self-test")
        run(cli)
