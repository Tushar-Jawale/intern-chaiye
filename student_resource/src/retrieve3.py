"""
Retrieval v3: for every Source 2/3 record, the top-K Source-1 records of the
same country label (a Source 2/3 record has at most one owner, so the search
runs from that side).

Keys come from the normalized fields: name words, name-word pairs, phonetic
skeletons, name/address character 4-grams, address words and pairs, 3+-digit
numbers, the whole number set, number|place composites, unit ids, a name
prefix, and an exact sorted-name bonus. A key is kept when at most max_df
Source-1 records carry it; its weight is kind weight x IDF. The score of a
pair is the summed weight of shared keys (sparse matrix product).

Keys are 64-bit hashes counted with numpy, so the index of a 1.3M-row country
fits in a few GB. On Linux the queries are split across forked workers.

    python src/retrieve3.py --work work3 --split train --top-k 10
    python src/retrieve3.py --work work3 --split train --dev-frac 0.1   # small world
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common3 import countries, dev_filter, load_norm, load_queries, owner_map, ret_path  # noqa: E402

KINDS = ["name", "npair", "phon", "addr", "apair", "num", "sig", "comp", "unit", "pref", "ngname", "ngaddr"]
# kind -> (max_df, weight). max_df None means scaled to the index size.
SPECS = {
    "name": (1500, 1.0),
    "npair": (8000, 2.4),
    "phon": (1500, 0.7),
    "addr": (1500, 0.9),
    "apair": (4000, 2.0),
    "num": (800, 1.2),
    "sig": (200, 2.8),
    "comp": (500, 2.2),
    "unit": (300, 2.6),
    "pref": (300, 0.8),
    "ngname": (None, 1.5),
    "ngaddr": (None, 1.1),
}
EXACT_BONUS = 40.0
EXACT_MAX_GROUP = 400
EXACT_SMALL_GROUP = 80
_KIND_ID = {k: i for i, k in enumerate(KINDS)}
_EX_KIND = len(KINDS)


def _pairs(words: list[str]) -> list[str]:
    uniq = sorted(set(words), key=len)
    uniq = sorted(uniq[-8:]) if len(uniq) > 8 else sorted(uniq)
    return [uniq[i] + "_" + uniq[j] for i in range(len(uniq)) for j in range(i + 1, len(uniq))]


def _ngrams(text: str, n: int = 4, limit: int = 12) -> list[str]:
    s = "".join(ch for ch in text if ch.isalnum())
    total = len(s) - n + 1
    if total <= 0:
        return []
    idxs = range(total) if total <= limit else [int(i * (total - 1) / (limit - 1)) for i in range(limit)]
    return list(dict.fromkeys(s[i:i + n] for i in idxs))


def rec_keys(core: str, phon: str, ad: str, nums: str, units: str) -> dict[str, list[str]]:
    words = [t for t in core.split() if len(t) >= 3 and not t.isdigit()]
    addr_words = [t for t in ad.split() if len(t) >= 3 and t.isalpha()]
    num_list = nums.split()
    longest = max(words, key=len) if words else ""
    sig = "#".join(sorted(set(num_list))) if len(set(num_list)) >= 2 else ""
    use_nums = [n for n in num_list if len(n) >= 2][:5]
    use_words = [w for w in addr_words if len(w) >= 4][:5]
    return {
        "name": words,
        "npair": _pairs(words),
        "phon": [p for p in phon.split() if len(p) >= 3],
        "addr": addr_words,
        "apair": _pairs(use_words + [w for w in addr_words if len(w) >= 4][5:8]),
        "num": [t for t in num_list if len(t) >= 3],
        "sig": [sig] if sig else [],
        "comp": [n + "|" + w for n in use_nums for w in use_words],
        "unit": units.split(),
        "pref": [longest[:5]] if len(longest) >= 5 else [],
        "ngname": _ngrams(core),
        "ngaddr": _ngrams(ad),
        "exact": [" ".join(sorted(set(words)))] if words else [],
    }


def _exact_keys(exact: str, sig: list[str], long_nums: list[str], df: int) -> list[str]:
    if not exact or df == 0 or df > EXACT_MAX_GROUP:
        return []
    if df <= EXACT_SMALL_GROUP:
        return ["ex|" + exact]
    keys = ["exs|" + exact + "|" + s for s in sig]
    keys.extend("exn|" + exact + "|" + t for t in long_nums)
    return keys


def _fields(df: pd.DataFrame) -> tuple[list[str], ...]:
    return (df["core"].tolist(), df["phon"].tolist(), df["ad"].tolist(), df["nums"].tolist(), df["units"].tolist())


def _emit(fields, start: int, end: int, exact_df: dict[int, int] | None):
    """Rows, key hashes and kind ids for records start..end."""
    core, phon, ad, nums, units = fields
    rows: list[int] = []
    hs: list[int] = []
    kd: list[int] = []
    for i in range(start, end):
        ks = rec_keys(core[i], phon[i], ad[i], nums[i], units[i])
        r = i - start
        for kind in KINDS:
            kid = _KIND_ID[kind]
            for key in ks[kind]:
                rows.append(r)
                hs.append(hash((kid, key)))
                kd.append(kid)
        if exact_df is not None and ks["exact"]:
            ex = ks["exact"][0]
            for key in _exact_keys(ex, ks["sig"], ks["num"], exact_df.get(hash(ex), 0)):
                rows.append(r)
                hs.append(hash((_EX_KIND, key)))
                kd.append(_EX_KIND)
    return (np.asarray(rows, np.int32), np.asarray(hs, np.int64), np.asarray(kd, np.int8))


class Index:
    def __init__(self, s1: pd.DataFrame):
        t0 = time.time()
        self.n = len(s1)
        fields = _fields(s1)
        ex_counts: dict[int, int] = {}
        for i in range(self.n):
            words = sorted(set(t for t in fields[0][i].split() if len(t) >= 3 and not t.isdigit()))
            if words:
                h = hash(" ".join(words))
                ex_counts[h] = ex_counts.get(h, 0) + 1
        self.exact_df = ex_counts
        parts = []
        step = 200_000
        for start in range(0, self.n, step):
            r, h, k = _emit(fields, start, min(start + step, self.n), ex_counts)
            parts.append((r + start, h, k))
        rows = np.concatenate([p[0] for p in parts])
        hs = np.concatenate([p[1] for p in parts])
        kd = np.concatenate([p[2] for p in parts])
        del parts
        # one (row, key) once
        order = np.lexsort((hs, rows))
        rows, hs, kd = rows[order], hs[order], kd[order]
        dup = np.zeros(len(rows), dtype=bool)
        dup[1:] = (rows[1:] == rows[:-1]) & (hs[1:] == hs[:-1])
        rows, hs, kd = rows[~dup], hs[~dup], kd[~dup]
        uniq, first, df = np.unique(hs, return_index=True, return_counts=True)
        ukind = kd[first]
        ng_cap = min(25000, max(80, self.n // 400))
        max_df = np.array([SPECS[k][0] if SPECS[k][0] is not None else ng_cap for k in KINDS] + [EXACT_MAX_GROUP], np.int64)
        base = np.array([SPECS[k][1] for k in KINDS] + [0.0], np.float32)
        keep = df <= max_df[ukind]
        weight = base[ukind] * np.log((self.n + 1) / df).astype(np.float32)
        weight[ukind == _EX_KIND] = EXACT_BONUS
        self.keys = uniq[keep]
        self.weight = weight[keep].astype(np.float32)
        col = np.searchsorted(self.keys, hs)
        col[col >= len(self.keys)] = 0
        ok = self.keys[col] == hs
        self.n_cols = len(self.keys)
        self.mat = sp.csr_matrix(
            (self.weight[col[ok]], (col[ok], rows[ok])), shape=(self.n_cols, self.n), dtype=np.float32
        )
        del rows, hs, kd, uniq, first, df, col, ok
        gc.collect()
        print(f"      index: {self.n:,} rows, {self.n_cols:,} keys, nnz {self.mat.nnz:,}, "
              f"4-gram cap {ng_cap:,} in {time.time() - t0:.0f}s", flush=True)

    def query(self, fields, start: int, end: int, top_k: int):
        r, h, _ = _emit(fields, start, end, self.exact_df)
        col = np.searchsorted(self.keys, h)
        col[col >= len(self.keys)] = 0
        ok = self.keys[col] == h
        q = sp.csr_matrix((np.ones(int(ok.sum()), np.float32), (r[ok], col[ok])), shape=(end - start, self.n_cols))
        q.data[:] = 1.0
        scores = (q @ self.mat).tocsr()
        out_q, out_s, out_r, out_v = [], [], [], []
        indptr, indices, data = scores.indptr, scores.indices, scores.data
        for row in range(end - start):
            a, b = indptr[row], indptr[row + 1]
            if a == b:
                continue
            d = data[a:b]
            idx = indices[a:b]
            if len(d) > top_k:
                part = np.argpartition(-d, top_k - 1)[:top_k]
                d, idx = d[part], idx[part]
            order = np.argsort(-d, kind="stable")
            m = len(order)
            out_q.append(np.full(m, start + row, np.int32))
            out_s.append(idx[order].astype(np.int32))
            out_r.append(np.arange(m, dtype=np.int8))
            out_v.append(d[order].astype(np.float32))
        if not out_q:
            e = np.empty(0, np.int32)
            return e, e, np.empty(0, np.int8), np.empty(0, np.float32)
        return np.concatenate(out_q), np.concatenate(out_s), np.concatenate(out_r), np.concatenate(out_v)


_G: dict = {}


def _work(span):
    start, end = span
    idx, fields, top_k, chunk = _G["index"], _G["fields"], _G["top_k"], _G["chunk"]
    parts = [idx.query(fields, s, min(s + chunk, end), top_k) for s in range(start, end, chunk)]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def search(index: Index, q: pd.DataFrame, top_k: int, chunk: int = 1000, workers: int = 1):
    fields = _fields(q)
    n = len(q)
    t0 = time.time()
    span = 50_000
    spans = [(s, min(s + span, n)) for s in range(0, n, span)]
    _G.update(index=index, fields=fields, top_k=top_k, chunk=chunk)
    results = []
    if workers > 1 and os.name == "posix":
        import multiprocessing as mp

        with mp.get_context("fork").Pool(workers) as pool:
            for i, res in enumerate(pool.imap(_work, spans)):
                results.append(res)
                if i % 10 == 0 or i == len(spans) - 1:
                    print(f"      queried {spans[i][1]:,}/{n:,} in {time.time() - t0:.0f}s", flush=True)
    else:
        for i, sp_ in enumerate(spans):
            results.append(_work(sp_))
            if i % 10 == 0 or i == len(spans) - 1:
                print(f"      queried {sp_[1]:,}/{n:,} in {time.time() - t0:.0f}s", flush=True)
    _G.clear()
    if not results:
        e = np.empty(0, np.int32)
        return e, e, np.empty(0, np.int8), np.empty(0, np.float32)
    return tuple(np.concatenate([r[i] for r in results]) for i in range(4))


def recall_report(q_ids, s_ids, qa, sa, ra, owner: dict[str, str], label: str) -> None:
    pos = {e: i for i, e in enumerate(s_ids)}
    own_idx = np.array([pos.get(owner.get(e, ""), -1) for e in q_ids], np.int64)
    owned = own_idx >= 0
    n_owned = int(owned.sum())
    hit_rank = np.full(len(q_ids), 99, np.int32)
    good = own_idx[qa] == sa
    np.minimum.at(hit_rank, qa[good], ra[good].astype(np.int32))
    parts = []
    for k in (1, 2, 3, 5, 10, 20):
        if k <= int(ra.max(initial=0)) + 1:
            parts.append(f"@{k}={np.mean(hit_rank[owned] < k):.4f}")
    print(f"  {label} owner recall ({n_owned:,} owned of {len(q_ids):,} queries): " + "  ".join(parts), flush=True)


def run(args: argparse.Namespace) -> None:
    os.makedirs(os.path.join(args.work, "ret"), exist_ok=True)
    split = args.split
    ctry_list = [args.country] if args.country else countries(args.work, split)
    workers = args.workers or (os.cpu_count() or 1)
    owner = owner_map(args.work) if split == "train" else {}
    for country in ctry_list:
        path = ret_path(args.work, split, country)
        if os.path.exists(path) and not args.force:
            print(f"skip {path}", flush=True)
            continue
        t0 = time.time()
        s1 = load_norm(args.work, f"{split}_s1", country)
        q = load_queries(args.work, split, country)
        if split == "train" and args.dev_frac < 1.0:
            s1, q = dev_filter(s1, q, owner, args.dev_frac)
        print(f"\n=== {split} {country}: S1={len(s1):,} queries={len(q):,} ===", flush=True)
        if len(s1) == 0 or len(q) == 0:
            continue
        index = Index(s1)
        qa, sa, ra, va = search(index, q, args.top_k, args.chunk, workers)
        del index
        gc.collect()
        q_ids = q["entity_id"].to_numpy().astype(str)
        s_ids = s1["entity_id"].to_numpy().astype(str)
        np.savez(path, q=qa, s=sa, rank=ra, score=va, q_ids=q_ids, s_ids=s_ids)
        print(f"  pairs {len(qa):,}  ({len(qa) / max(len(q), 1):.2f}/query) in {time.time() - t0:.0f}s -> {path}", flush=True)
        if owner:
            recall_report(q_ids, s_ids, qa, sa, ra, owner, country)


def self_test() -> None:
    s1 = pd.DataFrame({
        "entity_id": ["S1-a", "S1-b", "S1-c"],
        "core": ["alpha bakery", "zebra motors", "quiet lone"],
        "phon": ["lf bkr", "sbr mtrs", "kt ln"],
        "ad": ["100 main street phoenix", "9 oak road dallas", "1 empty lane"],
        "nums": ["100", "9", "1"],
        "units": ["", "", ""],
    })
    q = pd.DataFrame({
        "entity_id": ["S2-a", "S3-b", "S2-x"],
        "core": ["alpha bakrey", "zebra motorz", "unrelated pottery"],
        "phon": ["lf bkr", "sbr mtrs", "nrltd ptr"],
        "ad": ["100 main street phoenix", "9 oak road dallas", "50 other road"],
        "nums": ["100", "9", "50"],
        "units": ["", "", ""],
    })
    idx = Index(s1)
    qa, sa, ra, va = search(idx, q, 2, chunk=2, workers=1)
    top = {int(a): int(b) for a, b, r in zip(qa, sa, ra) if r == 0}
    assert top.get(0) == 0 and top.get(1) == 1, (qa, sa, ra, va)
    print("retrieve3 self-test OK", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval v3")
    p.add_argument("--work", default="")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--country", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--dev-frac", type=float, default=1.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    if cli.self_test:
        self_test()
    else:
        run(cli)
