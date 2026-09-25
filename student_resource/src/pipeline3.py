"""
v3 entity-resolution pipeline.

  prep      learn the Indic->Latin dictionary from train pairs, normalize all files
  retrieve  top-K Source-1 records for every Source 2/3 record (per country label)
  stage1    pair model on all K candidates. Train records are split in two folds;
            each fold's model scores the other fold, so every train probability
            is out-of-fold. Test gets the average of both fold models.
  stage2    top-M candidates per record by stage-1 probability, plus context:
            gap to the runner-up, how contested the Source-1 entity is, and how
            similar the record is to the other records already clustered on that
            entity. Ensemble (LightGBM / XGBoost / CatBoost / RF), blend weights
            and the decision rule are tuned on held-out entities.
  predict   test outputs: output/matching_results.tsv and output/candidate_pairs.tsv

Source-1 entities are split by a stable hash: report (never used for fitting or
tuning), tune (blend weights, calibration, decision rule), and train. The
REPORT line is computed like the leaderboard: macro F0.5 over every report
entity, singletons included, with every Source 2/3 record of the country in play.

    python src/pipeline3.py all --data dataset --work work3
    python src/pipeline3.py all --data dataset --work work3_dev --dev-frac 0.1 --skip-test
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decide3  # noqa: E402
import feats3  # noqa: E402
import models3  # noqa: E402
from common3 import bucket, countries, load_norm, load_queries, load_ret, owner_map, ret_path, safe  # noqa: E402

CTX_COLS = ["p1", "p1_rank", "p1_max_other", "p1_gap", "p1_sum_q", "p1_n_gt01",
            "s_psum_other", "s_pmax_other", "s_rank_of_q", "s_npairs"]
S1_FEATURES = feats3.FEATURES
S2_FEATURES = feats3.FEATURES + CTX_COLS + feats3.SIB_COLS
THR_GRID = [round(x, 3) for x in np.arange(0.10, 0.951, 0.025)] + [0.96, 0.97, 0.98, 0.99]


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------ context -------

class Ctx:
    """One split x country: retrieval pairs plus encoded records."""

    def __init__(self, work: str, split: str, country: str, need_tables: bool = True):
        t0 = time.time()
        r = load_ret(work, split, country)
        self.split, self.country = split, country
        self.qa = r["q"].astype(np.int32)
        self.sa = r["s"].astype(np.int32)
        self.rank = r["rank"].astype(np.int8)
        self.score = r["score"].astype(np.float32)
        self.q_ids = r["q_ids"].astype(object)
        self.s_ids = r["s_ids"].astype(object)
        self.nq, self.ns = len(self.q_ids), len(self.s_ids)
        self.q_start = np.searchsorted(self.qa, np.arange(self.nq), side="left")
        self.q_end = np.searchsorted(self.qa, np.arange(self.nq), side="right")
        if need_tables:
            s1 = load_norm(work, f"{split}_s1", country)
            q = load_queries(work, split, country)
            s1 = s1.set_index("entity_id").loc[list(self.s_ids)].reset_index()
            q = q.set_index("entity_id").loc[list(self.q_ids)].reset_index()
            self.TS, self.TQ, self.info = feats3.build_tables(s1, q)
            del s1, q
            gc.collect()
        log(f"  [{split} {country}] queries {self.nq:,}  S1 {self.ns:,}  pairs {len(self.qa):,}  ({time.time() - t0:.0f}s)")

    def chunks(self, size: int):
        n = len(self.qa)
        start = 0
        while start < n:
            end = min(start + size, n)
            if end < n:
                end = int(self.q_end[self.qa[end - 1]])
            yield start, end
            start = end

    def base_features(self, idx: np.ndarray, workers: int) -> np.ndarray:
        """Stage-1 features for ret pairs idx (sorted; whole queries or not)."""
        if len(idx) == 0:
            return np.zeros((0, len(S1_FEATURES)), np.float32)
        qa, sa = self.qa[idx], self.sa[idx]
        return np.hstack([
            feats3.kernel_features(self.TQ, self.TS, qa, sa, self.info),
            feats3.string_features(self.TQ, self.TS, qa, sa, workers),
            feats3.flag_features(self.TQ, self.TS, qa, sa),
            self.ret_features(idx),
        ]).astype(np.float32)

    def ret_features(self, idx: np.ndarray) -> np.ndarray:
        """Same columns as feats3.retrieval_features, for any subset of pairs."""
        qa = self.qa[idx]
        start, end = self.q_start[qa], self.q_end[qa]
        score = self.score[idx]
        best = self.score[start]
        size = (end - start).astype(np.float32)
        second = np.where(size > 1, self.score[np.minimum(start + 1, len(self.score) - 1)], 0.0)
        has_next = idx + 1 < end
        nxt = np.where(has_next, self.score[np.minimum(idx + 1, len(self.score) - 1)], 0.0)
        safe_best = np.maximum(best, 1e-6)
        return np.column_stack([
            self.rank[idx].astype(np.float32),
            np.log1p(score),
            np.where(best > 0, score / safe_best, 0.0),
            np.where(best > 0, (best - second) / safe_best, 0.0),
            size,
            np.where(score > 0, (score - nxt) / np.maximum(score, 1e-6), 0.0),
        ]).astype(np.float32)


def owner_index(ctx: Ctx, owner: dict[str, str]) -> np.ndarray:
    pos = {e: i for i, e in enumerate(ctx.s_ids)}
    return np.array([pos.get(owner.get(e, ""), -1) for e in ctx.q_ids], np.int32)


def entity_roles(s_ids, report_pct: int, tune_pct: int) -> np.ndarray:
    b = bucket(list(s_ids), 100)
    role = np.zeros(len(s_ids), np.int8)
    role[b < report_pct + tune_pct] = 1
    role[b < report_pct] = 2
    return role


# ------------------------------------------------------------- stage 1 ------

def _s1_prefix(work: str, fold: int) -> str:
    return os.path.join(work, "models", f"stage1_fold{fold}")


def _p1_path(work: str, split: str, country: str) -> str:
    return os.path.join(work, "p1", f"{split}_{safe(country)}.npy")


def _split_valid(q_of_row: np.ndarray, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    uq = np.unique(q_of_row)
    rng = np.random.default_rng(seed)
    val_q = rng.choice(uq, size=max(1, int(len(uq) * frac)), replace=False)
    return np.isin(q_of_row, val_q)


def stage1(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.work, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.work, "p1"), exist_ok=True)
    gpu = args.gpu
    kinds = args.s1_models.split(",")
    rng = np.random.default_rng(args.seed)
    ctry_train = countries_with_ret(args.work, "train")
    have_models = all(os.path.exists(_s1_prefix(args.work, f) + "_ensemble.json") for f in (0, 1))
    if not have_models or args.force:
        log("STAGE 1: building fit sample")
        owner = owner_map(args.work)
        xs = {0: [], 1: []}
        ys = {0: [], 1: []}
        qs = {0: [], 1: []}
        for country in ctry_train:
            ctx = Ctx(args.work, "train", country)
            own = owner_index(ctx, owner)
            role = entity_roles(ctx.s_ids, args.report_pct, args.tune_pct)
            q_eval = (own >= 0) & (role[np.maximum(own, 0)] > 0)
            fold = bucket(list(ctx.q_ids), 2)
            has = ctx.q_end > ctx.q_start
            for f in (0, 1):
                cand = np.flatnonzero(has & ~q_eval & (fold == f))
                take = rng.choice(cand, size=min(len(cand), args.s1_fit_queries), replace=False)
                sel = np.zeros(ctx.nq, dtype=bool)
                sel[take] = True
                idx = np.flatnonzero(sel[ctx.qa])
                for a, b in _ranges(len(idx), args.chunk_pairs):
                    part = idx[a:b]
                    xs[f].append(ctx.base_features(part, args.workers))
                    ys[f].append((own[ctx.qa[part]] == ctx.sa[part]).astype(np.int8))
                    qs[f].append(ctx.qa[part].astype(np.int64) + (hash(country) & 0xFFFF) * 10**8)
                log(f"    fold {f}: {len(take):,} queries, {len(idx):,} pairs")
            del ctx
            gc.collect()
        for f in (0, 1):
            x = np.vstack(xs[f])
            y = np.concatenate(ys[f])
            q = np.concatenate(qs[f])
            xs[f] = ys[f] = qs[f] = None
            val = _split_valid(q, 0.1, args.seed + f)
            log(f"  fold {f} model: {len(y):,} pairs, {int(y.sum()):,} positives")
            ens = models3.Ensemble(kinds)
            ens.fit(x[~val], y[~val], x[val], y[val], gpu, args.seed + f)
            ens.save(_s1_prefix(args.work, f))
            if f == 0:
                _print_gain(ens, S1_FEATURES)
            del x, y, q, ens
            gc.collect()
    models = {f: models3.Ensemble.load(_s1_prefix(args.work, f), gpu) for f in (0, 1)}
    owner = None
    for split in ("train", "test"):
        if split == "test" and args.skip_test:
            continue
        for country in countries_with_ret(args.work, split):
            path = _p1_path(args.work, split, country)
            if os.path.exists(path) and not args.force:
                log(f"  skip {path}")
                continue
            t0 = time.time()
            ctx = Ctx(args.work, split, country)
            p1 = np.zeros(len(ctx.qa), np.float32)
            if split == "train":
                owner = owner or owner_map(args.work)
                own = owner_index(ctx, owner)
                role = entity_roles(ctx.s_ids, args.report_pct, args.tune_pct)
                q_eval = (own >= 0) & (role[np.maximum(own, 0)] > 0)
                fold = bucket(list(ctx.q_ids), 2)
            for a, b in ctx.chunks(args.chunk_pairs):
                idx = np.arange(a, b)
                x = ctx.base_features(idx, args.workers)
                if split == "test":
                    p1[a:b] = 0.5 * (models[0].predict(x) + models[1].predict(x))
                else:
                    qa = ctx.qa[a:b]
                    ev = q_eval[qa]
                    use0 = (fold[qa] == 1) | ev
                    use1 = (fold[qa] == 0) | ev
                    p0 = np.zeros(b - a, np.float32)
                    pb = np.zeros(b - a, np.float32)
                    p0[use0] = models[0].predict(x[use0])
                    pb[use1] = models[1].predict(x[use1])
                    p1[a:b] = np.where(ev, 0.5 * (p0 + pb), np.where(fold[qa] == 0, pb, p0))
                log(f"    scored {b:,}/{len(ctx.qa):,} pairs ({time.time() - t0:.0f}s)")
            np.save(path, p1)
            if split == "train":
                _stage1_report(ctx, p1, own, q_eval)
            del ctx, p1
            gc.collect()


def _stage1_report(ctx: Ctx, p1: np.ndarray, own: np.ndarray, q_eval: np.ndarray) -> None:
    bq, bs, bp = decide3.best_per_query(ctx.qa, ctx.sa, p1)
    owned = own >= 0
    top_ok = np.zeros(ctx.nq, dtype=bool)
    top_ok[bq] = own[bq] == bs
    in_k = np.zeros(ctx.nq, dtype=bool)
    hit = own[ctx.qa] == ctx.sa
    in_k[ctx.qa[hit]] = True
    ret_top = np.zeros(ctx.nq, dtype=bool)
    r0 = ctx.rank == 0
    ret_top[ctx.qa[r0]] = own[ctx.qa[r0]] == ctx.sa[r0]
    log(f"  [{ctx.country}] owned queries: in top-K {in_k[owned].mean():.4f}  retrieval top-1 {ret_top[owned].mean():.4f}  "
        f"stage-1 top-1 {top_ok[owned].mean():.4f}")


def _print_gain(ens: models3.Ensemble, names: list[str]) -> None:
    m = ens.models.get("lgb")
    if m is None:
        return
    gain = m.feature_importance("gain")
    top = sorted(zip(names, gain), key=lambda kv: -kv[1])[:25]
    log("  top gain: " + ", ".join(f"{k}={v / max(gain.sum(), 1):.3f}" for k, v in top))


def _ranges(n: int, size: int):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def countries_with_ret(work: str, split: str) -> list[str]:
    return [c for c in countries(work, split) if os.path.exists(ret_path(work, split, c))]


# ------------------------------------------------------------- stage 2 ------

class P1Stats:
    """Per-query and per-Source-1 summaries of stage-1 probabilities."""

    def __init__(self, ctx: Ctx, p1: np.ndarray, top_m: int, tau: float):
        n = len(p1)
        order = np.lexsort((-p1, ctx.qa))
        qa_o = ctx.qa[order]
        first = np.ones(n, dtype=bool)
        first[1:] = qa_o[1:] != qa_o[:-1]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
        rank_o = np.arange(n) - grp_start
        self.p1_rank = np.empty(n, np.int16)
        self.p1_rank[order] = rank_o
        best = np.zeros(ctx.nq, np.float32)
        second = np.zeros(ctx.nq, np.float32)
        best_s = np.full(ctx.nq, -1, np.int32)
        r0 = order[rank_o == 0]
        best[ctx.qa[r0]] = p1[r0]
        best_s[ctx.qa[r0]] = ctx.sa[r0]
        r1 = order[rank_o == 1]
        second[ctx.qa[r1]] = p1[r1]
        self.q_best, self.q_second, self.best_s = best, second, best_s
        self.q_sum = np.bincount(ctx.qa, weights=p1, minlength=ctx.nq).astype(np.float32)
        self.q_n01 = np.bincount(ctx.qa, weights=(p1 > 0.1).astype(np.float64), minlength=ctx.nq).astype(np.float32)
        self.s_sum = np.bincount(ctx.sa, weights=p1, minlength=ctx.ns).astype(np.float32)
        self.s_n = np.bincount(ctx.sa, minlength=ctx.ns).astype(np.float32)
        so = np.lexsort((-p1, ctx.sa))
        sa_o = ctx.sa[so]
        sfirst = np.ones(n, dtype=bool)
        sfirst[1:] = sa_o[1:] != sa_o[:-1]
        sgs = np.maximum.accumulate(np.where(sfirst, np.arange(n), 0))
        srank_o = np.arange(n) - sgs
        self.s_rank = np.empty(n, np.int32)
        self.s_rank[so] = srank_o
        self.s_max1 = np.zeros(ctx.ns, np.float32)
        self.s_max2 = np.zeros(ctx.ns, np.float32)
        self.s_arg1 = np.full(ctx.ns, -1, np.int64)
        f0 = so[srank_o == 0]
        self.s_max1[ctx.sa[f0]] = p1[f0]
        self.s_arg1[ctx.sa[f0]] = f0
        f1 = so[srank_o == 1]
        self.s_max2[ctx.sa[f1]] = p1[f1]
        self.sel = np.sort(np.flatnonzero(self.p1_rank < top_m))
        self.tau = tau
        self.p1 = p1

    def ctx_features(self, ctx: Ctx, idx: np.ndarray) -> np.ndarray:
        p = self.p1[idx]
        qa, sa = ctx.qa[idx], ctx.sa[idx]
        is_top = self.p1_rank[idx] == 0
        max_other = np.where(is_top, self.q_second[qa], self.q_best[qa])
        s_other = np.where(self.s_arg1[sa] == idx, self.s_max2[sa], self.s_max1[sa])
        return np.column_stack([
            p, self.p1_rank[idx], max_other, p - max_other, self.q_sum[qa], self.q_n01[qa],
            self.s_sum[sa] - p, s_other, self.s_rank[idx], self.s_n[sa],
        ]).astype(np.float32)


def stage2_features(ctx: Ctx, st: P1Stats, idx: np.ndarray, workers: int) -> np.ndarray:
    base = ctx.base_features(idx, workers)
    cx = st.ctx_features(ctx, idx)
    sib = feats3.sibling_features(ctx.TQ, ctx.info, ctx.qa[idx], ctx.sa[idx], st.best_s, st.q_best, st.tau,
                                  workers=workers)
    return np.hstack([base, cx, sib]).astype(np.float32)


def _s2_data_path(work: str, country: str) -> str:
    return os.path.join(work, "s2data", f"train_{safe(country)}.npz")


def stage2(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.work, "s2data"), exist_ok=True)
    rng = np.random.default_rng(args.seed + 7)
    owner = owner_map(args.work)
    for country in countries_with_ret(args.work, "train"):
        path = _s2_data_path(args.work, country)
        if os.path.exists(path) and not args.force:
            log(f"  skip {path}")
            continue
        t0 = time.time()
        ctx = Ctx(args.work, "train", country)
        p1 = np.load(_p1_path(args.work, "train", country))
        st = P1Stats(ctx, p1, args.top_m, args.tau)
        own = owner_index(ctx, owner)
        role = entity_roles(ctx.s_ids, args.report_pct, args.tune_pct)
        sel = st.sel
        q_rel = np.zeros(ctx.nq, dtype=bool)
        q_rel[ctx.qa[sel[role[ctx.sa[sel]] > 0]]] = True
        q_rel |= (own >= 0) & (role[np.maximum(own, 0)] > 0)
        has = np.zeros(ctx.nq, dtype=bool)
        has[ctx.qa[sel]] = True
        cand = np.flatnonzero(has & ~q_rel)
        fitq = rng.choice(cand, size=min(len(cand), args.s2_fit_queries), replace=False)
        is_fit_q = np.zeros(ctx.nq, dtype=bool)
        is_fit_q[fitq] = True
        rows = sel[is_fit_q[ctx.qa[sel]] | q_rel[ctx.qa[sel]]]
        log(f"    stage-2 rows {len(rows):,}: fit queries {len(fitq):,}, eval-relevant queries {int(q_rel.sum()):,}")
        xs = []
        for a, b in _ranges(len(rows), args.chunk_pairs):
            xs.append(stage2_features(ctx, st, rows[a:b], args.workers))
        x = np.vstack(xs) if xs else np.zeros((0, len(S2_FEATURES)), np.float32)
        qa, sa = ctx.qa[rows], ctx.sa[rows]
        y = (own[qa] == sa).astype(np.int8)
        gold_ent = np.flatnonzero(role > 0)
        gold_q = [[] for _ in range(ctx.ns)]
        for qi in np.flatnonzero(own >= 0):
            if role[own[qi]] > 0:
                gold_q[own[qi]].append(qi)
        g_ent = np.concatenate([np.full(len(gold_q[e]), e, np.int32) for e in gold_ent]) if len(gold_ent) else np.zeros(0, np.int32)
        g_q = np.concatenate([np.asarray(gold_q[e], np.int32) for e in gold_ent]) if len(gold_ent) else np.zeros(0, np.int32)
        np.savez(path, x=x, y=y, qa=qa, sa=sa, fit=is_fit_q[qa], rel=q_rel[qa], role_s=role, gold_ent=gold_ent,
                 g_ent=g_ent, g_q=g_q, q_ids=ctx.q_ids.astype(str), s_ids=ctx.s_ids.astype(str))
        in_m = np.zeros(ctx.nq, dtype=bool)
        in_m[ctx.qa[sel][own[ctx.qa[sel]] == ctx.sa[sel]]] = True
        owned = own >= 0
        log(f"  [{country}] owned queries with owner in top-{args.top_m}: {in_m[owned].mean():.4f}  "
            f"-> {path} ({time.time() - t0:.0f}s)")
        del ctx, p1, st, x, xs
        gc.collect()
    train_stage2(args)


def _gold(d, which: int) -> dict[str, set[str]]:
    role = d["role_s"]
    s_ids, q_ids = d["s_ids"], d["q_ids"]
    gold = {s_ids[e]: set() for e in d["gold_ent"] if role[e] == which}
    for e, q in zip(d["g_ent"], d["g_q"]):
        if role[e] == which:
            gold[s_ids[e]].add(q_ids[q])
    return gold


def train_stage2(args: argparse.Namespace) -> None:
    prefix = os.path.join(args.work, "models", "stage2")
    meta_path = os.path.join(args.work, "models", "stage2_meta.json")
    if os.path.exists(meta_path) and not args.force:
        log(f"  skip stage-2 training ({meta_path} exists)")
        return
    data = {c: dict(np.load(_s2_data_path(args.work, c), allow_pickle=False)) for c in countries_with_ret(args.work, "train")}
    xf = np.vstack([d["x"][d["fit"]] for d in data.values()])
    yf = np.concatenate([d["y"][d["fit"]] for d in data.values()])
    qf = np.concatenate([d["qa"][d["fit"]].astype(np.int64) + i * 10**9 for i, d in enumerate(data.values())])
    val = _split_valid(qf, 0.1, args.seed + 11)
    log(f"STAGE 2: fit rows {len(yf):,} positives {int(yf.sum()):,}")
    ens = models3.Ensemble(args.s2_models.split(","))
    ens.fit(xf[~val], yf[~val], xf[val], yf[val], args.gpu, args.seed)
    del xf, yf, qf
    gc.collect()
    _print_gain(ens, S2_FEATURES)
    # predictions for eval-relevant rows
    parts_all = {}
    for c, d in data.items():
        m = d["rel"]
        parts_all[c] = ens.predict_each(d["x"][m])
    # blend weights on rows touching tune entities only
    tune_parts = {k: [] for k in ens.kinds}
    tune_y = []
    for c, d in data.items():
        m = d["rel"]
        tune_rows = d["role_s"][d["sa"][m]] == 1
        for k in ens.kinds:
            tune_parts[k].append(parts_all[c][k][tune_rows])
        tune_y.append(d["y"][m][tune_rows])
    tune_parts = {k: np.concatenate(v) for k, v in tune_parts.items()}
    tune_y = np.concatenate(tune_y)
    ens.tune_weights(tune_parts, tune_y)
    ens.save(prefix)
    # best pair per query and calibration
    best = {}
    cal_p, cal_y = [], []
    for c, d in data.items():
        m = d["rel"]
        p = ens.blend(parts_all[c])
        qa, sa, yy = d["qa"][m], d["sa"][m], d["y"][m]
        order = np.lexsort((-p, qa))
        first = np.ones(len(order), dtype=bool)
        first[1:] = qa[order][1:] != qa[order][:-1]
        rows = order[first]
        best[c] = (qa[rows], sa[rows], p[rows])
        tm = d["role_s"][sa[rows]] == 1
        cal_p.append(p[rows][tm])
        cal_y.append(yy[rows][tm].astype(np.float64))
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.concatenate(cal_p), np.concatenate(cal_y))
    calib = lambda v: iso.predict(v).astype(np.float64)  # noqa: E731

    def evaluate(which: int, method: str, **kw) -> tuple[float, dict]:
        preds: dict[str, set[str]] = {}
        gold: dict[str, set[str]] = {}
        per_c = {}
        for c, d in data.items():
            bq, bs, bp = best[c]
            mask = decide3.select(bq, bs, bp, method, calib=calib if method == "ef" else None, **kw)
            pr = decide3.to_pred(bq, bs, mask, d["q_ids"], d["s_ids"])
            g = _gold(d, which)
            pr = {k: v for k, v in pr.items() if k in g}
            per_c[c] = decide3.macro_report(pr, g)
            preds.update(pr)
            gold.update(g)
        rep = decide3.macro_report(preds, gold)
        rep["countries"] = {c: (v["f05"], v["n"]) for c, v in per_c.items()}
        return rep["f05"], rep

    trials = []
    for t in THR_GRID:
        f, _ = evaluate(1, "thr", thr=t)
        trials.append((f, "thr", {"thr": t}))
    for floor in (0.02, 0.05, 0.1, 0.2, 0.3):
        for extra in (0.0, 0.25, 0.5):
            f, _ = evaluate(1, "ef", floor=floor, extra=extra)
            trials.append((f, "ef", {"floor": floor, "extra": extra}))
    trials.sort(key=lambda t: -t[0])
    best_thr = max((t for t in trials if t[1] == "thr"), key=lambda t: t[0])
    best_ef = max((t for t in trials if t[1] == "ef"), key=lambda t: t[0])
    log(f"  tune: best thr {best_thr[2]} F0.5={best_thr[0]:.4f} | best ef {best_ef[2]} F0.5={best_ef[0]:.4f}")
    f_star, method, kw = trials[0]
    for label, (_, mth, k2) in (("thr", best_thr), ("ef", best_ef)):
        _, rep = evaluate(2, mth, **k2)
        log(f"  report[{label}]: F0.5={rep['f05']:.4f}  P={rep['precision']:.4f}  R={rep['recall']:.4f}  "
            f"singleton FP={rep['singleton_fp']:.3f}  n={rep['n']:,}")
    _, rep = evaluate(2, method, **kw)
    log(f"REPORT MACRO F0.5 = {rep['f05']:.4f}  (method {method} {kw}, P={rep['precision']:.4f} R={rep['recall']:.4f} "
        f"singleton FP={rep['singleton_fp']:.3f})")
    for k, (f, n) in rep["countries"].items():
        log(f"    country={k:8s} F0.5={f:.4f}  n={n:,}")
    for k, (f, n) in rep["groups"].items():
        log(f"    {k:12s} F0.5={f:.4f}  n={n:,}")
    meta = {
        "method": method, "params": kw, "tune_f05": f_star, "report": {k: v for k, v in rep.items()},
        "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
        "features": S2_FEATURES, "top_m": args.top_m, "tau": args.tau,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=float)
    log(f"Wrote {meta_path}")


# ------------------------------------------------------------- predict ------

def predict(args: argparse.Namespace) -> None:
    meta_path = os.path.join(args.work, "models", "stage2_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    ens = models3.Ensemble.load(os.path.join(args.work, "models", "stage2"), args.gpu)
    iso_x, iso_y = np.asarray(meta["iso_x"]), np.asarray(meta["iso_y"])
    calib = lambda v: np.interp(v, iso_x, iso_y)  # noqa: E731
    method, kw = meta["method"], meta["params"]
    top_m, tau = int(meta["top_m"]), float(meta["tau"])
    matches: dict[str, list[str]] = {}
    cands: dict[str, list[str]] = {}
    for country in countries_with_ret(args.work, "test"):
        t0 = time.time()
        ctx = Ctx(args.work, "test", country)
        p1 = np.load(_p1_path(args.work, "test", country))
        st = P1Stats(ctx, p1, top_m, tau)
        sel = st.sel
        p2 = np.zeros(len(sel), np.float32)
        for a, b in _ranges(len(sel), args.chunk_pairs):
            p2[a:b] = ens.predict(stage2_features(ctx, st, sel[a:b], args.workers))
            log(f"    stage-2 scored {b:,}/{len(sel):,} ({time.time() - t0:.0f}s)")
        bq, bs, bp = decide3.best_per_query(ctx.qa[sel], ctx.sa[sel], p2)
        mask = decide3.select(bq, bs, bp, method, calib=calib if method == "ef" else None, **kw)
        for qi, si in zip(bq[mask].tolist(), bs[mask].tolist()):
            matches.setdefault(ctx.s_ids[si], []).append(ctx.q_ids[qi])
        for qi, si in zip(ctx.qa[sel].tolist(), ctx.sa[sel].tolist()):
            cands.setdefault(ctx.s_ids[si], []).append(ctx.q_ids[qi])
        log(f"  [test {country}] matched records {int(mask.sum()):,} of {ctx.nq:,} ({time.time() - t0:.0f}s)")
        del ctx, p1, st
        gc.collect()
    s1_all = load_norm(args.work, "test_s1", None, ["entity_id"])["entity_id"].tolist()
    out = args.output or os.path.join(args.work, "output")
    os.makedirs(out, exist_ok=True)
    _write(os.path.join(out, "matching_results.tsv"), "matched_entity_ids", matches, s1_all)
    _write(os.path.join(out, "candidate_pairs.tsv"), "candidate_entity_ids", cands, s1_all)
    n_match = sum(len(v) for v in matches.values())
    log(f"Wrote {out}/matching_results.tsv ({n_match:,} matches over {len(matches):,} entities) and candidate_pairs.tsv")


def _write(path: str, col: str, rows: dict[str, list[str]], order: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"source1_entity_id\t{col}\n")
        for eid in order:
            ids = list(dict.fromkeys(rows.get(eid, [])))
            handle.write(eid + "\t" + ",".join(ids) + "\n")


# ----------------------------------------------------------------- main -----

def prep(args: argparse.Namespace) -> None:
    import norm3

    if not os.path.exists(os.path.join(args.work, "translit.json")) or args.force:
        norm3.learn_cmd(argparse.Namespace(data=args.data, work=args.work))
    splits = "train" if args.skip_test else "train,test"
    norm3.build_cmd(argparse.Namespace(data=args.data, work=args.work, workers=args.workers if args.workers > 0 else 0,
                                       splits=splits, force=args.force))


def retrieve(args: argparse.Namespace) -> None:
    import retrieve3

    for split in ("train", "test"):
        if split == "test" and args.skip_test:
            continue
        retrieve3.run(argparse.Namespace(work=args.work, split=split, country="", top_k=args.top_k, chunk=1000,
                                         workers=args.workers if args.workers > 0 else 0,
                                         dev_frac=args.dev_frac if split == "train" else 1.0, force=args.force))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3 entity-resolution pipeline")
    p.add_argument("cmd", choices=["prep", "retrieve", "stage1", "stage2", "train2", "predict", "all"])
    p.add_argument("--data", default="dataset")
    p.add_argument("--work", default="work3")
    p.add_argument("--output", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-m", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.5, help="stage-1 probability that puts a record in a cluster")
    p.add_argument("--dev-frac", type=float, default=1.0, help="train only: hash slice of the world for quick runs")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--report-pct", type=int, default=4)
    p.add_argument("--tune-pct", type=int, default=4)
    p.add_argument("--s1-models", default="", help="default xgb on GPU, lgb otherwise")
    p.add_argument("--s2-models", default="", help="default lgb,xgb,cat")
    p.add_argument("--s1-fit-queries", type=int, default=300_000, help="per fold per country")
    p.add_argument("--s2-fit-queries", type=int, default=600_000, help="per country")
    p.add_argument("--chunk-pairs", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=-1)
    p.add_argument("--gpu", default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    a.gpu = models3.gpu_available() if a.gpu == "auto" else a.gpu == "yes"
    if not a.s1_models:
        a.s1_models = "xgb" if a.gpu else "lgb"
    if not a.s2_models:
        a.s2_models = "lgb,xgb,cat"
    return a


def main() -> None:
    args = parse_args()
    log(f"GPU: {args.gpu}  stage-1 models: {args.s1_models}  stage-2 models: {args.s2_models}")
    t0 = time.time()
    if args.cmd in ("prep", "all"):
        prep(args)
    if args.cmd in ("retrieve", "all"):
        retrieve(args)
    if args.cmd in ("stage1", "all"):
        stage1(args)
    if args.cmd in ("stage2", "all"):
        stage2(args)
    if args.cmd == "train2":
        train_stage2(args)
    if args.cmd in ("predict", "all") and not args.skip_test:
        predict(args)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
