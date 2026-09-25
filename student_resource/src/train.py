"""
Phase 4 — Train the matcher on blocking candidates.

Labels come from train ground truth. A candidate that is a true match is a
positive. Every other candidate of that entity is a hard negative. By default
ALL candidates of the training entities are used (neg_per_pos=0) with no class
weighting, so predicted probabilities live on the same scale the model sees at
inference (about 1 positive per 60 candidates).

Entities are split three ways before any feature is computed:
  fit        -> model training
  threshold  -> pick the decision threshold that maximises macro F0.5
  report     -> the number we quote (never used for fitting or tuning)

    python src/train.py --preprocessed preprocessed --candidates output/sample_candidate_pairs.tsv --output output
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
        if keep_all or neg_per_pos <= 0:
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


THRESHOLD_GRID = sorted(
    set([round(float(t), 3) for t in np.linspace(0.30, 0.95, 27)])
    | {0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999}
)


def entity_report(preds: dict[str, set[str]], gold: dict[str, set[str]], countries: dict[str, str] | None = None) -> dict:
    """Macro F0.5 / precision / recall, plus F0.5 by true-match count and country."""
    f_sum = p_sum = r_sum = 0.0
    groups: dict[str, list[float]] = {}
    singleton_fp = singleton_n = 0
    for eid, truth in gold.items():
        pred = preds.get(eid) or set()
        if not truth:
            singleton_n += 1
            score = 1.0 if not pred else 0.0
            singleton_fp += 1 if pred else 0
            precision = recall = score
        elif not pred:
            score = precision = recall = 0.0
        else:
            hit = len(truth & pred)
            precision = hit / len(pred)
            recall = hit / len(truth)
            score = (1.25 * precision * recall) / (0.25 * precision + recall) if hit else 0.0
        f_sum += score
        p_sum += precision
        r_sum += recall
        n_true = len(truth)
        key = "matches=0" if n_true == 0 else "matches=1" if n_true == 1 else "matches=2-3" if n_true <= 3 else "matches=4+"
        groups.setdefault(key, []).append(score)
        if countries is not None:
            groups.setdefault("country=" + str(countries.get(eid, "?")), []).append(score)
    n = max(len(gold), 1)
    return {
        "macro_f05": f_sum / n,
        "macro_precision": p_sum / n,
        "macro_recall": r_sum / n,
        "singleton_false_positive_rate": singleton_fp / max(singleton_n, 1),
        "groups": {k: {"f05": float(np.mean(v)), "n": len(v)} for k, v in sorted(groups.items())},
    }


def best_threshold(s1_ids: list[str], cand_ids: list[str], scores: np.ndarray, gold: dict[str, set[str]]) -> tuple[float, float, list]:
    entities = {eid: gold.get(eid, set()) for eid in set(s1_ids)}
    best_t, best_f = 0.5, -1.0
    curve = []
    for threshold in THRESHOLD_GRID:
        preds = predictions_at(s1_ids, cand_ids, scores, float(threshold))
        for eid in entities:
            preds.setdefault(eid, set())
        rep = entity_report(preds, entities)
        curve.append((float(threshold), rep["macro_f05"], rep["macro_precision"], rep["macro_recall"]))
        if rep["macro_f05"] > best_f:
            best_t, best_f = float(threshold), rep["macro_f05"]
    return best_t, best_f, curve


def load_countries(preprocessed: str, entity_ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    path = os.path.join(preprocessed, "train_s1.parquet")
    if not os.path.exists(path):
        return out
    pf = pq.ParquetFile(path)
    if "country" not in pf.schema_arrow.names:
        return out
    for batch in pf.iter_batches(batch_size=250_000, columns=["entity_id", "country"]):
        data = batch.to_pydict()
        for eid, country in zip(data["entity_id"], data["country"]):
            if eid in entity_ids:
                out[eid] = country
    return out


def run(args: argparse.Namespace) -> None:
    import lightgbm as lgb

    os.makedirs(args.output, exist_ok=True)
    print("Reading candidates", flush=True)
    candidates = read_candidates(args.candidates, args.max_entities, args.seed)
    entity_ids = list(candidates)
    rng = random.Random(args.seed)
    rng.shuffle(entity_ids)
    n_ent = len(entity_ids)
    cut_fit = max(1, int(n_ent * 0.70))
    cut_thr = max(cut_fit + 1, int(n_ent * 0.85))
    fit_ids, thr_ids, rep_ids = entity_ids[:cut_fit], entity_ids[cut_fit:cut_thr], entity_ids[cut_thr:]
    print(f"  fit entities {len(fit_ids):,}  threshold entities {len(thr_ids):,}  report entities {len(rep_ids):,}", flush=True)

    gold = load_gold(args.preprocessed, set(entity_ids))
    countries = load_countries(args.preprocessed, set(entity_ids))
    need = set(entity_ids)
    for eid in entity_ids:
        need.update(candidates[eid])
    print(f"Loading {len(need):,} records", flush=True)
    t0 = time.time()
    records = load_records(args.preprocessed, need)
    print(f"  loaded {len(records):,} in {time.time() - t0:.0f}s", flush=True)

    print("Building fit pairs", flush=True)
    x_fit, y_fit, _, _ = build_rows(candidates, gold, records, fit_ids, args.neg_per_pos, args.seed, keep_all=False)
    print(f"  fit pairs {len(y_fit):,}  positives {int(y_fit.sum()):,}  ({y_fit.mean() if len(y_fit) else 0:.4f})", flush=True)
    x_thr, y_thr, thr_s1, thr_cands = build_rows(candidates, gold, records, thr_ids, 0, args.seed, keep_all=True)
    x_rep, y_rep, rep_s1, rep_cands = build_rows(candidates, gold, records, rep_ids, 0, args.seed, keep_all=True)
    print(f"  threshold pairs {len(y_thr):,}  report pairs {len(y_rep):,}", flush=True)
    if len(y_fit) == 0 or len(y_thr) == 0 or len(y_rep) == 0:
        raise RuntimeError("No pairs built. Check the candidate file and preprocessed path.")

    pos = max(int(y_fit.sum()), 1)
    neg = max(len(y_fit) - pos, 1)
    scale_pos_weight = (neg / pos) if args.class_weight else 1.0
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_fit, y_fit)

    thr_scores = model.predict_proba(x_thr)[:, 1]
    threshold, f05_thr, curve = best_threshold(thr_s1, thr_cands, thr_scores, gold)
    print("Threshold sweep (threshold split): thr  F0.5  P  R", flush=True)
    for t, f, p, r in curve:
        marker = " <- best" if t == threshold else ""
        print(f"  {t:.3f}  {f:.4f}  {p:.4f}  {r:.4f}{marker}", flush=True)
    if threshold >= max(THRESHOLD_GRID):
        print("  WARNING: best threshold is the top of the grid; probabilities are still inflated.", flush=True)

    rep_scores = model.predict_proba(x_rep)[:, 1]
    rep_gold = {eid: gold.get(eid, set()) for eid in rep_ids}
    rep_preds = predictions_at(rep_s1, rep_cands, rep_scores, threshold)
    for eid in rep_ids:
        rep_preds.setdefault(eid, set())
    report = entity_report(rep_preds, rep_gold, countries)
    print(f"REPORT MACRO F0.5: {report['macro_f05']:.4f}  P={report['macro_precision']:.4f}  R={report['macro_recall']:.4f}  "
          f"singleton FP rate={report['singleton_false_positive_rate']:.3f}  at threshold {threshold:.3f}", flush=True)
    for key, val in report["groups"].items():
        print(f"  {key:14s} F0.5={val['f05']:.4f}  n={val['n']:,}", flush=True)
    importance = sorted(zip(FEATURE_COLS, model.booster_.feature_importance("gain")), key=lambda kv: -kv[1])
    print("  feature gain:", [(k, int(v)) for k, v in importance], flush=True)

    model_path = os.path.join(args.output, "matcher.txt")
    model.booster_.save_model(model_path)
    meta = {
        "threshold": threshold,
        "threshold_split_macro_f05": f05_thr,
        "report_split": report,
        "threshold_curve": curve,
        "features": FEATURE_COLS,
        "n_fit_entities": len(fit_ids),
        "neg_per_pos": args.neg_per_pos,
        "class_weight": bool(args.class_weight),
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "matcher_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Wrote {model_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the matching model")
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-entities", type=int, default=100000)
    p.add_argument("--neg-per-pos", type=int, default=0, help="0 = use every candidate as a negative (default)")
    p.add_argument("--class-weight", action="store_true", help="Re-enable scale_pos_weight (inflates probabilities)")
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
