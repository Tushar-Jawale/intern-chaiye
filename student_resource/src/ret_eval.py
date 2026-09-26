"""
Retrieval recall at full index size on a sample of owned queries.

Builds the real Source-1 index of one train country (every record), queries a
random sample of owned Source 2/3 records, and reports owner recall@k plus, for
misses, which key kinds the owner shared with the query and whether those keys
were dropped by the frequency cap.

    python src/ret_eval.py --work work3 --country India --n-queries 50000 --top-k 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import retrieve3  # noqa: E402
from common3 import load_norm, load_queries, owner_map  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="work3")
    p.add_argument("--country", default="India")
    p.add_argument("--n-queries", type=int, default=50_000)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diag", type=int, default=3000, help="misses to diagnose")
    a = p.parse_args()

    t0 = time.time()
    owner = owner_map(a.work)
    s1 = load_norm(a.work, "train_s1", a.country)
    q = load_queries(a.work, "train", a.country)
    q = q[q["entity_id"].map(lambda e: e in owner)].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    q = q.iloc[np.sort(rng.choice(len(q), size=min(a.n_queries, len(q)), replace=False))].reset_index(drop=True)
    print(f"S1 {len(s1):,}  sampled owned queries {len(q):,}  ({time.time() - t0:.0f}s)", flush=True)

    index = retrieve3.Index(s1)
    t1 = time.time()
    qa, sa, ra, va = retrieve3.search(index, q, a.top_k, chunk=1000, workers=1)
    dt = time.time() - t1
    print(f"query time {dt:.0f}s  ({1000 * dt / len(q):.2f} ms/query, {len(qa) / len(q):.1f} cands/query)", flush=True)

    pos = {e: i for i, e in enumerate(s1["entity_id"].tolist())}
    own = np.array([pos.get(owner[e], -1) for e in q["entity_id"]], np.int64)
    hit_rank = np.full(len(q), 999, np.int32)
    good = own[qa] == sa
    np.minimum.at(hit_rank, qa[good], ra[good].astype(np.int32))
    ks = [k for k in (1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50) if k <= a.top_k]
    print("owner recall: " + "  ".join(f"@{k}={np.mean(hit_rank < k):.4f}" for k in ks), flush=True)

    miss = np.flatnonzero(hit_rank >= 10)[: a.diag]
    fq = retrieve3._fields(q)
    fs = retrieve3._fields(s1)
    kept = set(index.keys.tolist())
    shared_kinds = Counter()
    shared_kept = Counter()
    none_shared = 0
    only_capped = 0
    for i in miss:
        o = own[i]
        kq = retrieve3.rec_keys(*(f[i] for f in fq))
        ko = retrieve3.rec_keys(*(f[o] for f in fs))
        any_shared = any_kept = False
        for kind in retrieve3.KINDS:
            common = set(kq[kind]) & set(ko[kind])
            if not common:
                continue
            any_shared = True
            shared_kinds[kind] += 1
            if any(hash((retrieve3._KIND_ID[kind], k)) in kept for k in common):
                shared_kept[kind] += 1
                any_kept = True
        none_shared += not any_shared
        only_capped += any_shared and not any_kept
    n = max(len(miss), 1)
    print(f"misses@10 diagnosed {len(miss):,}: no shared key {none_shared / n:.3f}  "
          f"shared keys all capped {only_capped / n:.3f}", flush=True)
    for kind in retrieve3.KINDS:
        print(f"   {kind:7s} shared {shared_kinds[kind] / n:.3f}  kept {shared_kept[kind] / n:.3f}", flush=True)
    for i in miss[:12]:
        o = own[i]
        print(f"   Q: {fq[0][i]} | {fq[2][i]}\n   S: {fs[0][o]} | {fs[2][o]}  rank={hit_rank[i]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
