"""
Name-difference vocabulary learned from train retrieval pairs.

For candidate pairs that share at least one name word, the words present in
the Source 2/3 name but not in the Source-1 name ("added") and the reverse
("dropped") are counted separately for true and false pairs. The data has
strongly asymmetric vocabularies: variants of the same business add words like
"formerly", "fka", "center", "service", "www", while look-alike records of a
different business add words like "industries", "ventures", "holdings",
"group". The smoothed log-odds per word become pair features (feats3).

Pairs touching report/tune Source-1 entities are left out, so the reported
score never sees its own labels through this table.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import numpy as np

from common3 import bucket, countries, load_norm, load_queries, load_ret, owner_map, ret_path, safe

MIN_COUNT = 20
CLIP = 8.0


def table_path(work: str) -> str:
    return os.path.join(work, "diff_table.json")


def learn(work: str, report_pct: int, tune_pct: int, max_pairs: int = 4_000_000, top_rank: int = 5,
          seed: int = 0) -> dict:
    t0 = time.time()
    owner = owner_map(work)
    rng = np.random.default_rng(seed)
    add = {True: Counter(), False: Counter()}
    drop = {True: Counter(), False: Counter()}
    n = {True: 0, False: 0}
    for country in countries(work, "train"):
        if not os.path.exists(ret_path(work, "train", country)):
            continue
        r = load_ret(work, "train", country)
        s_ids, q_ids = r["s_ids"].astype(str), r["q_ids"].astype(str)
        role = bucket(list(s_ids), 100) < report_pct + tune_pct
        s_pos = {e: i for i, e in enumerate(s_ids)}
        own = np.array([s_pos.get(owner.get(e, ""), -1) for e in q_ids], np.int64)
        q_eval = (own >= 0) & role[np.maximum(own, 0)]
        m = (r["rank"] < top_rank) & ~role[r["s"]] & ~q_eval[r["q"]]
        idx = np.flatnonzero(m)
        if len(idx) > max_pairs:
            idx = np.sort(rng.choice(idx, size=max_pairs, replace=False))
        qa, sa = r["q"][idx], r["s"][idx]
        s_core = load_norm(work, "train_s1", country, ["entity_id", "core"]).set_index("entity_id")["core"]
        q_core = load_queries(work, "train", country, ["entity_id", "core"]).set_index("entity_id")["core"]
        s_core = s_core.reindex(s_ids).fillna("").to_numpy()
        q_core = q_core.reindex(q_ids).fillna("").to_numpy()
        y = own[qa] == sa
        for a, b, lab in zip(q_core[qa].tolist(), s_core[sa].tolist(), y.tolist()):
            A = set(a.split())
            B = set(b.split())
            if not (A & B):
                continue
            n[lab] += 1
            add[lab].update(A - B)
            drop[lab].update(B - A)
        print(f"  diff table: {country} pairs {len(idx):,} ({time.time() - t0:.0f}s)", flush=True)

    def lo(cp: Counter, cn: Counter) -> dict[str, float]:
        out = {}
        npos, nneg = max(n[True], 1), max(n[False], 1)
        for t in set(cp) | set(cn):
            a, b = cp[t], cn[t]
            if a + b < MIN_COUNT:
                continue
            v = math.log((a + 0.5) / npos) - math.log((b + 0.5) / nneg)
            out[t] = float(max(-CLIP, min(CLIP, v)))
        return out

    table = {"add": lo(add[True], add[False]), "drop": lo(drop[True], drop[False]),
             "n_pos": n[True], "n_neg": n[False]}
    with open(table_path(work), "w", encoding="utf-8") as h:
        json.dump(table, h)
    worst = sorted(table["add"].items(), key=lambda kv: kv[1])[:8]
    best = sorted(table["add"].items(), key=lambda kv: -kv[1])[:8]
    print(f"  diff table: {len(table['add']):,} added / {len(table['drop']):,} dropped words from "
          f"{n[True]:,} true and {n[False]:,} false pairs ({time.time() - t0:.0f}s)", flush=True)
    print(f"    distractor-like: {worst}\n    match-like: {best}", flush=True)
    return table


def load(work: str) -> dict | None:
    path = table_path(work)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as h:
        return json.load(h)


# ---------------------------------------------------------------- proxy ----
# The supervised table only knows train vocabulary; France has its own
# distractor words (participations, holding, distribution, ...). Look-alike
# records of another business also change the house number while variants of
# the same business keep it, so "is the Source-1 number kept when this word is
# added/dropped" is a label-free stand-in for the label. It is learned per
# split x country from that country's own retrieval pairs, so it exists in the
# same form for every country, France included.

def proxy_path(work: str, split: str, country: str) -> str:
    return os.path.join(work, "diff", f"proxy_{split}_{safe(country)}.json")


def learn_proxy(work: str, split: str, country: str, max_pairs: int = 4_000_000, top_rank: int = 3,
                seed: int = 0) -> dict:
    t0 = time.time()
    r = load_ret(work, split, country)
    idx = np.flatnonzero(r["rank"] < top_rank)
    if len(idx) > max_pairs:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_pairs, replace=False))
    s_ids, q_ids = r["s_ids"].astype(str), r["q_ids"].astype(str)
    S = load_norm(work, f"{split}_s1", country, ["entity_id", "core", "nums"]).set_index("entity_id")
    Q = load_queries(work, split, country, ["entity_id", "core", "nums"]).set_index("entity_id")
    S = S.reindex(s_ids).fillna("")
    Q = Q.reindex(q_ids).fillna("")
    sc, sn, qc, qn = S["core"].tolist(), S["nums"].tolist(), Q["core"].tolist(), Q["nums"].tolist()
    add = {True: Counter(), False: Counter()}
    drop = {True: Counter(), False: Counter()}
    for a, b in zip(r["q"][idx].tolist(), r["s"][idx].tolist()):
        A, B = set(qc[a].split()), set(sc[b].split())
        if not (A & B):
            continue
        na, nb = set(qn[a].split()), set(sn[b].split())
        if not na or not nb:
            continue
        kept = nb <= na
        plus, minus = A - B, B - A
        if plus and len(minus) <= 1:
            add[kept].update(plus)
        if minus and len(plus) <= 1:
            drop[kept].update(minus)

    def lo(side: dict) -> dict[str, float]:
        nk, nc = max(sum(side[True].values()), 1), max(sum(side[False].values()), 1)
        out = {}
        for t in set(side[True]) | set(side[False]):
            k, c = side[True][t], side[False][t]
            if k + c >= MIN_COUNT:
                out[t] = float(max(-CLIP, min(CLIP, math.log((k + 0.5) / nk) - math.log((c + 0.5) / nc))))
        return out

    table = {"add": lo(add), "drop": lo(drop)}
    os.makedirs(os.path.dirname(proxy_path(work, split, country)), exist_ok=True)
    with open(proxy_path(work, split, country), "w", encoding="utf-8") as h:
        json.dump(table, h)
    print(f"  proxy table {split} {country}: {len(table['add']):,} added / {len(table['drop']):,} dropped words "
          f"from {len(idx):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return table


def load_proxy(work: str, split: str, country: str) -> dict | None:
    path = proxy_path(work, split, country)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as h:
        return json.load(h)
