"""
Phase 4 — Train the matcher on blocking candidates.

Labels come from train ground truth. A candidate that is a true match is a
positive. Every other candidate of that entity is a hard negative. Entities
are split before any feature is computed.

The decision threshold maximises macro F0.5 on the validation entities,
including singletons. That is the leaderboard metric.

Kaggle, after the sample candidate file exists:

    from argparse import Namespace
    import glob, os
    hits = glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)
    args = Namespace(
        preprocessed=os.path.dirname(hits[0]),
        candidates="/kaggle/working/output/sample_candidate_pairs.tsv",
        output="/kaggle/working/output",
        max_entities=20000,
        neg_per_pos=4,
        seed=42,
    )
    run(args)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from features import FEATURE_COLS, macro_f05, pair_features
except ImportError:
    # A notebook cell that already ran features.py defines these names.
    pass

COLS = ["entity_id", "name_core", "addr_normalized", "addr_numbers"]


def read_candidates(path: str, max_entities: int, seed: int) -> dict[str, list[str]]:
    """Reservoir-sample Source-1 rows so a full candidate file stays in memory."""
    rng = random.Random(seed)
    kept: list[tuple[str, list[str]]] = []
    seen = 0
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
            seen += 1
            row = (eid, cands)
            if len(kept) < max_entities:
                kept.append(row)
            else:
                j = rng.randrange(seen)
                if j < max_entities:
                    kept[j] = row
    print(f"  sampled {len(kept):,} entities from {seen:,} candidate rows", flush=True)
    return dict(kept)


def load_gold(preprocessed: str, entity_ids: set[str]) -> dict[str, set[str]]:
    gt = pd.read_parquet(
        os.path.join(preprocessed, "train_gt.parquet"),
        columns=["source1_entity_id", "matched_entity_ids"],
    )
    gold: dict[str, set[str]] = {}
    for eid, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if eid not in entity_ids:
            continue
        if matched is None or (isinstance(matched, float) and pd.isna(matched)):
            gold[eid] = set()
            continue
        text = str(matched).strip()
        if not text or text == "nan":
            gold[eid] = set()
        else:
            gold[eid] = {m.strip() for m in text.split(",") if m.strip()}
    for eid in entity_ids:
        gold.setdefault(eid, set())
    return gold


def load_records(preprocessed: str, need: set[str], names: tuple[str, ...] | None = None) -> dict[str, tuple[str, str, str]]:
    records: dict[str, tuple[str, str, str]] = {}
    file_names = names or ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3")
    for name in file_names:
        path = os.path.join(preprocessed, name + ".parquet")
        if not os.path.exists(path):
            continue
        pf = pq.ParquetFile(path)
        present = [c for c in COLS if c in pf.schema_arrow.names]
        if "entity_id" not in present:
            continue
        for batch in pf.iter_batches(batch_size=250_000, columns=present):
            data = batch.to_pydict()
            ids = data["entity_id"]
            name_col = data.get("name_core", [""] * len(ids))
            addrs = data.get("addr_normalized", [""] * len(ids))
            nums = data.get("addr_numbers", [""] * len(ids))
            for i, eid in enumerate(ids):
                if eid in need and eid not in records:
                    records[eid] = (name_col[i] or "", addrs[i] or "", nums[i] or "")
        if need <= records.keys():
            break
    missing = len(need) - len(need & records.keys())
    if missing:
        print(f"  records missing: {missing:,}", flush=True)
    return records


def build_rows(
    candidates: dict[str, list[str]],
    gold: dict[str, set[str]],
    records: dict[str, tuple[str, str, str]],
    entity_ids: list[str],
    neg_per_pos: int,
    seed: int,
    keep_all: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    s1_ids: list[str] = []
    cand_ids: list[str] = []
    empty = ("", "", "")
    for eid in entity_ids:
        cands = candidates.get(eid) or []
        truth = gold.get(eid) or set()
        left = records.get(eid, empty)
        chosen = []
        if keep_all:
            chosen = cands
        else:
            pos = [c for c in cands if c in truth]
            neg = [c for c in cands if c not in truth]
            n_neg = max(neg_per_pos, neg_per_pos * max(len(pos), 1))
            if len(neg) > n_neg:
                pick = rng.choice(len(neg), size=n_neg, replace=False)
                neg = [neg[i] for i in pick]
            chosen = pos + neg
        for cid in chosen:
            right = records.get(cid)
            if right is None:
                continue
            xs.append(pair_features(left[0], left[1], left[2], right[0], right[1], right[2]))
            ys.append(1 if cid in truth else 0)
            s1_ids.append(eid)
            cand_ids.append(cid)
    if not xs:
        return np.zeros((0, len(FEATURE_COLS))), np.zeros(0, dtype=np.int8), [], []
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8), s1_ids, cand_ids


def predictions_at(s1_ids: list[str], cand_ids: list[str], scores: np.ndarray, threshold: float) -> dict[str, set[str]]:
    preds: dict[str, set[str]] = {}
    for eid, cid, score in zip(s1_ids, cand_ids, scores):
        bucket = preds.setdefault(eid, set())
        if score >= threshold:
            bucket.add(cid)
    return preds


def best_threshold(s1_ids: list[str], cand_ids: list[str], scores: np.ndarray, gold: dict[str, set[str]]) -> tuple[float, float]:
    entities = {eid: gold.get(eid, set()) for eid in set(s1_ids)}
    best_t, best_f = 0.5, -1.0
    for threshold in np.linspace(0.30, 0.95, 27):
        preds = predictions_at(s1_ids, cand_ids, scores, float(threshold))
        for eid in entities:
            preds.setdefault(eid, set())
        score = macro_f05(preds, entities)
        if score > best_f:
            best_t, best_f = float(threshold), score
    return best_t, best_f


def run(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    os.makedirs(args.output, exist_ok=True)
    print("Reading candidates", flush=True)
    candidates = read_candidates(args.candidates, args.max_entities, args.seed)
    entity_ids = list(candidates)
    rng = random.Random(args.seed)
    rng.shuffle(entity_ids)
    cut = max(1, int(len(entity_ids) * 0.8))
    train_ids, val_ids = entity_ids[:cut], entity_ids[cut:]
    print(f"  train entities {len(train_ids):,}  val entities {len(val_ids):,}", flush=True)

    gold = load_gold(args.preprocessed, set(entity_ids))
    need = set(entity_ids)
    for eid in entity_ids:
        need.update(candidates[eid])
    print(f"Loading {len(need):,} records", flush=True)
    t0 = time.time()
    records = load_records(args.preprocessed, need)
    print(f"  loaded {len(records):,} in {time.time() - t0:.0f}s", flush=True)

    print("Building train pairs", flush=True)
    x_train, y_train, _, _ = build_rows(
        candidates, gold, records, train_ids, args.neg_per_pos, args.seed, keep_all=False
    )
    print(f"  train pairs {len(y_train):,}  positives {int(y_train.sum()):,}", flush=True)
    print("Scoring every validation candidate", flush=True)
    x_val, y_val, val_s1, val_cands = build_rows(
        candidates, gold, records, val_ids, args.neg_per_pos, args.seed, keep_all=True
    )
    print(f"  val pairs {len(y_val):,}  positives {int(y_val.sum()):,}", flush=True)
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError("No pairs built. Check the candidate file and preprocessed path.")

    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="binary_logloss")
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold, f05 = best_threshold(val_s1, val_cands, val_scores, gold)
    print(f"VAL MACRO F0.5: {f05:.4f} at threshold {threshold:.3f}", flush=True)

    model_path = os.path.join(args.output, "matcher.txt")
    model.booster_.save_model(model_path)
    meta = {"threshold": threshold, "val_macro_f05": f05, "features": FEATURE_COLS}
    with open(os.path.join(args.output, "matcher_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Wrote {model_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the matching model")
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-entities", type=int, default=20000)
    p.add_argument("--neg-per-pos", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
