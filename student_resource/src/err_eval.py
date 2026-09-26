"""
Full-density error analysis of retrieval + stage 1 on one train country.

The real Source-1 index of the country (every record, or the test-like world
with --orphan-frac) is queried with a sample of Source 2/3 records; stage-1
fold models from --models score the candidates. Reports top-1 accuracy of owned
records, the confidence given to unowned ones, and error examples by type.

    python src/err_eval.py --work work3 --models work3_dev2/models --country India --n-owned 30000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diff3  # noqa: E402
import feats3  # noqa: E402
import models3  # noqa: E402
import retrieve3  # noqa: E402
from common3 import bucket, load_norm, load_queries, orphan_filter, owner_map  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="work3")
    p.add_argument("--models", default="work3_dev2/models")
    p.add_argument("--country", default="India")
    p.add_argument("--n-owned", type=int, default=30_000)
    p.add_argument("--n-unowned", type=int, default=10_000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--orphan-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-pct", type=int, default=4)
    p.add_argument("--tune-pct", type=int, default=4)
    p.add_argument("--out", default="")
    a = p.parse_args()

    t0 = time.time()
    owner = owner_map(a.work)
    s1 = orphan_filter(load_norm(a.work, "train_s1", a.country), a.orphan_frac)
    in_world = set(s1["entity_id"])
    q = load_queries(a.work, "train", a.country)
    own_all = q["entity_id"].map(lambda e: owner.get(e, ""))
    owned = own_all.map(lambda o: o in in_world).to_numpy()
    rng = np.random.default_rng(a.seed)
    i_own = rng.choice(np.flatnonzero(owned), size=min(a.n_owned, int(owned.sum())), replace=False)
    i_un = rng.choice(np.flatnonzero(~owned), size=min(a.n_unowned, int((~owned).sum())), replace=False)
    q = q.iloc[np.sort(np.concatenate([i_own, i_un]))].reset_index(drop=True)
    print(f"S1 {len(s1):,}  queries {len(q):,}  ({time.time() - t0:.0f}s)", flush=True)

    index = retrieve3.Index(s1)
    qa, sa, ra, va = retrieve3.search(index, q, a.top_k, chunk=1000, workers=1)
    del index
    ens = [models3.Ensemble.load(os.path.join(a.models, f"stage1_fold{f}"), False) for f in (0, 1)]
    model_work = os.path.dirname(os.path.normpath(a.models))
    table = diff3.load(model_work)
    use_diff = table is not None
    feats3.set_diff_table(table)
    TS, TQ, info = feats3.build_tables(s1, q)
    feats3.attach_proxy(info, diff3.load_proxy(model_work, "train", a.country))
    parts = [
        feats3.kernel_features(TQ, TS, qa, sa, info),
        feats3.string_features(TQ, TS, qa, sa, -1),
        feats3.flag_features(TQ, TS, qa, sa),
        feats3.retrieval_features(qa, ra, va),
    ]
    if use_diff:
        parts.append(feats3.diff_features(TQ, TS, qa, sa, info))
    x = np.hstack(parts).astype(np.float32)
    p1 = 0.5 * (ens[0].predict(x) + ens[1].predict(x))
    print(f"pairs {len(qa):,} scored ({time.time() - t0:.0f}s)", flush=True)

    pos = {e: i for i, e in enumerate(s1["entity_id"].tolist())}
    own = np.array([pos.get(owner.get(e, ""), -1) for e in q["entity_id"]], np.int64)
    order = np.lexsort((-p1, qa))
    first = np.ones(len(order), bool)
    first[1:] = qa[order][1:] != qa[order][:-1]
    bq, bs, bp = qa[order][first], sa[order][first], p1[order][first]
    second = np.zeros(len(q), np.float32)
    rank_in = np.empty(len(order), np.int32)
    grp = np.cumsum(first) - 1
    rank_in[order] = np.arange(len(order)) - np.flatnonzero(first)[grp]
    sec_rows = order[rank_in[order] == 1]
    second[qa[sec_rows]] = p1[sec_rows]
    best_s = np.full(len(q), -1)
    best_p = np.zeros(len(q), np.float32)
    best_s[bq] = bs
    best_p[bq] = bp
    in_k = np.zeros(len(q), bool)
    hit = own[qa] == sa
    in_k[qa[hit]] = True
    p_own = np.zeros(len(q), np.float32)
    p_own[qa[hit]] = p1[hit]
    isown = own >= 0
    top_ok = isown & (best_s == own)
    print(f"owned: in top-{a.top_k} {in_k[isown].mean():.4f}  stage-1 top-1 {top_ok[isown].mean():.4f}", flush=True)
    for t in (0.3, 0.5, 0.7, 0.9):
        print(f"  thr {t}: owned correct&kept {np.mean(top_ok[isown] & (best_p[isown] >= t)):.4f}  "
              f"owned wrong&kept {np.mean(~top_ok[isown] & (best_p[isown] >= t)):.4f}  "
              f"unowned kept {np.mean(best_p[~isown] >= t):.4f}", flush=True)
    # Held out from the diff table: owned records of report/tune entities, and
    # unowned records whose best candidate is a report/tune entity.
    role = bucket(s1["entity_id"].tolist(), 100) < a.report_pct + a.tune_pct
    h_own = isown & role[np.maximum(own, 0)]
    h_un = ~isown & (best_s >= 0) & role[np.maximum(best_s, 0)]
    orphan = np.array([owner.get(e, "") != "" for e in q["entity_id"]]) & ~isown
    print(f"held-out: owned {int(h_own.sum()):,} top-1 {top_ok[h_own].mean():.4f}  unowned {int(h_un.sum()):,}", flush=True)
    for t in (0.3, 0.5, 0.7, 0.9):
        print(f"  thr {t}: owned correct&kept {np.mean(top_ok[h_own] & (best_p[h_own] >= t)):.4f}  "
              f"unowned kept {np.mean(best_p[h_un] >= t):.4f}  "
              f"(orphans {np.mean(best_p[h_un & orphan] >= t):.4f} / natural {np.mean(best_p[h_un & ~orphan] >= t):.4f})",
              flush=True)

    cols = ["core", "ad", "nums"]
    Q = {c: q[c].tolist() for c in cols}
    S = {c: s1[c].tolist() for c in cols}
    lines = []

    def show(tag, i, s_alt):
        o = own[i]
        lines.append(f"[{tag}] best_p={best_p[i]:.3f} p_owner={p_own[i]:.3f} second={second[i]:.3f}")
        lines.append(f"   Q : {Q['core'][i]} | {Q['ad'][i]}")
        if o >= 0:
            lines.append(f"   O : {S['core'][o]} | {S['ad'][o]}")
        if s_alt >= 0 and s_alt != o:
            lines.append(f"   B : {S['core'][s_alt]} | {S['ad'][s_alt]}")

    miss = np.flatnonzero(isown & ~in_k)
    outr = np.flatnonzero(isown & in_k & ~top_ok)
    low = np.flatnonzero(top_ok & (best_p < 0.5))
    fp = np.flatnonzero(~isown & (best_p >= 0.5))
    print(f"errors: not retrieved {len(miss):,}  outranked {len(outr):,}  right but p<0.5 {len(low):,}  "
          f"unowned with p>=0.5 {len(fp):,}", flush=True)
    for tag, arr in (("NOT-RETRIEVED", miss), ("OUTRANKED", outr), ("LOW-P", low), ("UNOWNED-KEPT", fp)):
        for i in rng.permutation(arr)[:40]:
            show(tag, i, best_s[i])
    out = a.out or os.path.join(a.work, f"err_{a.country}.txt")
    with open(out, "w", encoding="utf-8") as h:
        h.write("\n".join(lines))
    print(f"examples -> {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
