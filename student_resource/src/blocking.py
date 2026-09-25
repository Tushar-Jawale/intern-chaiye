"""
Phase 2 — Blocking / candidate generation.

Measured on the full train split (2.2M Source-1 rows, 10.3M candidates):
  candidate recall 0.951 overall (India 0.910, US 0.978) at top_k=200.

Each country label found in the data is blocked separately (France on test is
just another label). Keys per record: name words, name-word pairs, address
words, address-word pairs, 3+-digit numbers, the whole address-number set,
number|place composites, a 5-letter name prefix, and the exact sorted name.
A key is kept only when it is shared by at most max_df candidates. Score is
the sum of IDF weights of shared keys; each Source-1 row keeps the top-K.

Scoring is a sparse matrix product (Source-1 keys x candidate keys), computed
in chunks of Source-1 rows with scipy. Same keys and weights as the earlier
Python posting-list loop, which took ~4 hours per split on one core.

    python src/blocking.py --mode sample --sample-size 100000   # train sample + recall
    python src/blocking.py --mode test                          # candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

COLS = [
    "entity_id",
    "country",
    "name_core",
    "addr_normalized",
    "addr_numbers",
]


def _words(text: str, min_len: int) -> list[str]:
    if not text:
        return []
    return [t for t in str(text).split() if len(t) >= min_len and not t.isdigit()]


def _all_numbers(text: str) -> list[str]:
    """Every digit token, including short ones like 6 and 29."""
    if not text:
        return []
    return [t for t in str(text).split() if t.isdigit() and t != "0"]


def _signature(nums: list[str]) -> list[str]:
    """One key for the whole number set. Short numbers are useless alone."""
    usable = sorted(set(nums))
    if len(usable) < 2:
        return []
    return ["#".join(usable)]


def _composites(nums: list[str], addr_words: list[str]) -> list[str]:
    """Number + place word. Catches a typo'd name at the same premises."""
    keys = []
    use_nums = [n for n in nums if len(n) >= 2][:5]
    use_words = [w for w in addr_words if len(w) >= 4][:5]
    for num in use_nums:
        for word in use_words:
            keys.append(num + "|" + word)
    return keys


def _pairs(words: list[str]) -> list[str]:
    """Sorted word pairs. A pair stays rare when each word is common."""
    uniq = sorted(set(words), key=len)
    if len(uniq) > 8:
        uniq = sorted(uniq[-8:])
    else:
        uniq = sorted(uniq)
    if len(uniq) < 2:
        return []
    return [uniq[i] + "_" + uniq[j] for i in range(len(uniq)) for j in range(i + 1, len(uniq))]


def _prefix(name_words: list[str]) -> str:
    if not name_words:
        return ""
    longest = max(name_words, key=len)
    return longest[:5] if len(longest) >= 5 else ""


def _idf(df: int, n: int) -> float:
    return math.log((n + 1) / df)


# kind -> (max_df, weight multiplier). Common single words are not keys by
# themselves; they survive inside a pair, which is rare.
_SPECS = {
    "name": (1500, 1.0),
    "npair": (8000, 2.4),
    "addr": (1500, 0.9),
    "apair": (4000, 2.0),
    "num": (800, 1.2),
    "sig": (200, 2.8),
    "comp": (500, 2.2),
    "pref": (300, 0.8),
}
_EXACT_BONUS = 40.0
_EXACT_MAX_GROUP = 400
_EXACT_SMALL_GROUP = 80


def record_keys(name: str, addr: str, nums_text: str) -> dict[str, list[str]]:
    """All blocking keys of one record, grouped by kind. Same on both sides."""
    words = _words(name, 3)
    addr_words = _words(addr, 3)
    nums = _all_numbers(nums_text)
    pref = _prefix(words)
    return {
        "name": words,
        "npair": _pairs(words),
        "addr": addr_words,
        "apair": _pairs([w for w in addr_words if len(w) >= 4]),
        "num": [t for t in nums if len(t) >= 3],
        "sig": _signature(nums),
        "comp": _composites(nums, addr_words),
        "pref": [pref] if pref else [],
        "exact": [" ".join(sorted(set(words)))] if words else [],
    }


def _exact_keys(exact: str, sig: list[str], long_nums: list[str], df: int) -> list[str]:
    """Exact sorted-name bonus. Small groups match outright; large groups only
    when the address numbers agree (whole set, or a shared 3+-digit number)."""
    if not exact or df == 0 or df > _EXACT_MAX_GROUP:
        return []
    if df <= _EXACT_SMALL_GROUP:
        return ["ex|" + exact]
    keys = []
    if sig:
        keys.append("exs|" + exact + "|" + sig[0])
    for t in long_nums:
        keys.append("exn|" + exact + "|" + t)
    return keys


def _topk_rows(scores: sp.csr_matrix, top_k: int, min_score: float) -> list[np.ndarray]:
    """Per row: column ids of the top_k largest scores >= min_score, best first."""
    out = []
    indptr, indices, data = scores.indptr, scores.indices, scores.data
    for r in range(scores.shape[0]):
        a, b = indptr[r], indptr[r + 1]
        if a == b:
            out.append(np.empty(0, dtype=np.int64))
            continue
        d = data[a:b]
        idx = indices[a:b]
        if len(d) > top_k:
            part = np.argpartition(-d, top_k - 1)[:top_k]
            d, idx = d[part], idx[part]
        order = np.argsort(-d, kind="stable")
        d, idx = d[order], idx[order]
        out.append(idx[d >= min_score])
    return out


def _coo_from_buffers(rows: list[int], cols: list[int], vals: list[float], shape: tuple[int, int]) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.asarray(vals, dtype=np.float32), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=shape,
        dtype=np.float32,
    )


def block_frames(
    s1: pd.DataFrame,
    cand: pd.DataFrame,
    top_k: int = 200,
    min_score: float = 1.0,
    chunk: int = 2000,
) -> dict[str, list[str]]:
    """Block one country. Returns source1 id -> candidate ids, best first."""
    s1_ids = s1["entity_id"].tolist()
    s1_name = s1["name_core"].fillna("").astype(str).tolist()
    s1_addr = s1["addr_normalized"].fillna("").astype(str).tolist()
    s1_nums = s1["addr_numbers"].fillna("").astype(str).tolist()

    c_ids = cand["entity_id"].tolist()
    c_name = cand["name_core"].fillna("").astype(str).tolist()
    c_addr = cand["addr_normalized"].fillna("").astype(str).tolist()
    c_nums = cand["addr_numbers"].fillna("").astype(str).tolist()
    n = len(c_ids)
    if n == 0 or len(s1_ids) == 0:
        return {eid: [] for eid in s1_ids}

    # Pass 1: document frequencies per kind (candidate side only, label-free).
    t0 = time.time()
    print(f"      counting keys on {n:,} candidates", flush=True)
    counts: dict[str, Counter] = {k: Counter() for k in _SPECS}
    exact_counts: Counter = Counter()
    for i in range(n):
        ks = record_keys(c_name[i], c_addr[i], c_nums[i])
        for kind in _SPECS:
            if ks[kind]:
                counts[kind].update(set(ks[kind]))
        if ks["exact"]:
            exact_counts[ks["exact"][0]] += 1

    col: dict[str, int] = {}
    weights: list[float] = []
    for kind, (max_df, base) in _SPECS.items():
        for key, c in counts[kind].items():
            if 1 <= c <= max_df:
                col[kind + "|" + key] = len(weights)
                weights.append(base * _idf(c, n))
    del counts
    gc.collect()

    # Pass 2: candidate matrix (rows = candidates, cols = keys, value = weight).
    print(f"      building candidate matrix ({len(col):,} keys) at {time.time() - t0:.0f}s", flush=True)
    exact_col: dict[str, int] = {}
    buf_r: list[int] = []
    buf_c: list[int] = []
    buf_v: list[float] = []
    arr_r: list[np.ndarray] = []
    arr_c: list[np.ndarray] = []
    arr_v: list[np.ndarray] = []
    n_base = len(weights)

    def flush_buffers() -> None:
        if buf_r:
            arr_r.append(np.asarray(buf_r, dtype=np.int32))
            arr_c.append(np.asarray(buf_c, dtype=np.int32))
            arr_v.append(np.asarray(buf_v, dtype=np.float32))
            buf_r.clear()
            buf_c.clear()
            buf_v.clear()

    for i in range(n):
        ks = record_keys(c_name[i], c_addr[i], c_nums[i])
        seen: set[int] = set()
        for kind in _SPECS:
            for key in ks[kind]:
                j = col.get(kind + "|" + key)
                if j is not None and j not in seen:
                    seen.add(j)
                    buf_r.append(i)
                    buf_c.append(j)
                    buf_v.append(weights[j])
        if ks["exact"]:
            ex = ks["exact"][0]
            for key in _exact_keys(ex, ks["sig"], ks["num"], exact_counts[ex]):
                j = exact_col.get(key)
                if j is None:
                    j = n_base + len(exact_col)
                    exact_col[key] = j
                buf_r.append(i)
                buf_c.append(j)
                buf_v.append(_EXACT_BONUS)
        if len(buf_r) >= 2_000_000:
            flush_buffers()
    flush_buffers()
    del c_name, c_addr, c_nums
    n_cols = n_base + len(exact_col)
    rows_a = np.concatenate(arr_r)
    cols_a = np.concatenate(arr_c)
    vals_a = np.concatenate(arr_v)
    del arr_r, arr_c, arr_v
    # keys x candidates, built directly in the orientation used for scoring
    cand_t = sp.csr_matrix((vals_a, (cols_a, rows_a)), shape=(n_cols, n), dtype=np.float32)
    del rows_a, cols_a, vals_a
    gc.collect()
    print(f"      matrix nnz={cand_t.nnz:,} at {time.time() - t0:.0f}s", flush=True)

    # Pass 3: score Source-1 rows in chunks. Query matrix is binary.
    n_s1 = len(s1_ids)
    out: dict[str, list[str]] = {}
    t1 = time.time()
    n_chunks = (n_s1 + chunk - 1) // chunk
    for ci, start in enumerate(range(0, n_s1, chunk)):
        end = min(start + chunk, n_s1)
        q_r: list[int] = []
        q_c: list[int] = []
        for r, i in enumerate(range(start, end)):
            ks = record_keys(s1_name[i], s1_addr[i], s1_nums[i])
            seen = set()
            for kind in _SPECS:
                for key in ks[kind]:
                    j = col.get(kind + "|" + key)
                    if j is not None and j not in seen:
                        seen.add(j)
                        q_r.append(r)
                        q_c.append(j)
            if ks["exact"]:
                ex = ks["exact"][0]
                for key in _exact_keys(ex, ks["sig"], ks["num"], exact_counts.get(ex, 0)):
                    j = exact_col.get(key)
                    if j is not None and j not in seen:
                        seen.add(j)
                        q_r.append(r)
                        q_c.append(j)
        q = _coo_from_buffers(q_r, q_c, [1.0] * len(q_r), (end - start, n_cols))
        scores = (q @ cand_t).tocsr()
        for r, cols in enumerate(_topk_rows(scores, top_k, min_score)):
            out[s1_ids[start + r]] = [c_ids[j] for j in cols.tolist()]
        del q, scores
        if (ci + 1) % 25 == 0 or end == n_s1:
            print(f"      scored {end:,}/{n_s1:,} in {time.time() - t1:.0f}s", flush=True)
    del cand_t
    gc.collect()
    return out


def find_preprocessed(explicit: str | None) -> str:
    if explicit:
        return explicit
    src = globals().get("__file__")
    here = os.path.dirname(os.path.abspath(src)) if src else os.getcwd()
    candidates = [
        os.path.join(here, "..", "preprocessed"),
        os.path.join(os.getcwd(), "preprocessed"),
    ]
    env = os.environ.get("PREPROCESSED_DIR")
    if env:
        candidates.insert(0, env)
    kaggle_root = "/kaggle/input"
    if os.path.isdir(kaggle_root):
        candidates.extend(glob.glob(os.path.join(kaggle_root, "*")))
        candidates.extend(glob.glob(os.path.join(kaggle_root, "*", "*")))
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "train_s1.parquet")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find train_s1.parquet. Pass --preprocessed, or set PREPROCESSED_DIR."
    )


def default_output() -> str:
    if os.path.isdir("/kaggle/working"):
        path = "/kaggle/working/output"
    else:
        src = globals().get("__file__")
        here = os.path.dirname(os.path.abspath(src)) if src else os.getcwd()
        path = os.path.join(here, "..", "output")
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def load_split(preprocessed: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    s1 = pd.read_parquet(os.path.join(preprocessed, f"{split}_s1.parquet"), columns=COLS)
    s2 = pd.read_parquet(os.path.join(preprocessed, f"{split}_s2.parquet"), columns=COLS)
    s3 = pd.read_parquet(os.path.join(preprocessed, f"{split}_s3.parquet"), columns=COLS)
    gt = None
    if split == "train":
        gt = pd.read_parquet(
            os.path.join(preprocessed, "train_gt.parquet"),
            columns=["source1_entity_id", "matched_entity_ids"],
        )
    return s1, pd.concat([s2, s3], ignore_index=True), gt


def recall_against(candidates: dict[str, list[str]], gt: pd.DataFrame) -> tuple[int, int, list, Counter]:
    """Pair recall plus a histogram of the rank at which true matches were found."""
    found = 0
    total = 0
    missed = []
    ranks: Counter = Counter()
    pos_of = {k: {c: i for i, c in enumerate(v)} for k, v in candidates.items()}
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s1_id not in pos_of:
            continue
        if matched is None or (isinstance(matched, float) and pd.isna(matched)):
            continue
        text = str(matched).strip()
        if not text or text == "nan":
            continue
        got = pos_of[s1_id]
        for mid in text.split(","):
            mid = mid.strip()
            if not mid:
                continue
            total += 1
            r = got.get(mid)
            if r is not None:
                found += 1
                ranks["<10" if r < 10 else "<25" if r < 25 else "<50" if r < 50 else "<100" if r < 100 else ">=100"] += 1
            elif len(missed) < 15:
                missed.append((s1_id, mid))
    return found, total, missed, ranks


def candidate_stats(rows: dict[str, list[str]], top_k: int) -> str:
    sizes = np.fromiter((len(v) for v in rows.values()), dtype=np.int64, count=len(rows))
    if len(sizes) == 0:
        return "no rows"
    return (
        f"avg={sizes.mean():.1f} median={np.median(sizes):.0f} zero={(sizes == 0).sum():,} "
        f"at_top_k={(sizes >= top_k).mean():.3f}"
    )


def write_tsv(path: str, rows: dict[str, list[str]], ordered_ids: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in ordered_ids:
            seen = set()
            uniq = []
            for c in rows.get(eid) or []:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            f.write(eid + "\t" + ",".join(uniq) + "\n")


def run(args: argparse.Namespace) -> None:
    preprocessed = find_preprocessed(args.preprocessed)
    out_dir = args.output or default_output()
    split = "test" if args.mode == "test" else "train"
    chunk = getattr(args, "chunk", 2000) or 2000
    print(f"Preprocessed: {preprocessed}", flush=True)
    print(f"Output:       {out_dir}", flush=True)
    t_load = time.time()
    s1, cand, gt = load_split(preprocessed, split)
    print(
        f"Loaded {split}: S1={len(s1):,} candidates={len(cand):,} in {time.time() - t_load:.0f}s",
        flush=True,
    )

    if args.mode == "sample":
        s1 = s1.sample(n=min(args.sample_size, len(s1)), random_state=42).reset_index(drop=True)
        print(f"Sample mode: scoring {len(s1):,} Source-1 rows (indexes use ALL candidates)", flush=True)

    if args.country:
        countries = [args.country]
    else:
        countries = sorted(s1["country"].dropna().unique().tolist())

    all_rows: dict[str, list[str]] = {}
    found = total = 0
    missed_all = []
    ranks_all: Counter = Counter()
    t0 = time.time()
    for country in countries:
        s1_c = s1[s1["country"] == country].reset_index(drop=True)
        cand_c = cand[cand["country"] == country].reset_index(drop=True)
        print(
            f"\n=== {country}: S1={len(s1_c):,} candidates={len(cand_c):,} ===",
            flush=True,
        )
        if len(s1_c) == 0:
            continue
        t_c = time.time()
        blocked = block_frames(s1_c, cand_c, top_k=args.top_k, min_score=args.min_score, chunk=chunk)
        all_rows.update(blocked)
        print(
            f"  {country} done in {time.time() - t_c:.0f}s, candidates/entity {candidate_stats(blocked, args.top_k)}",
            flush=True,
        )
        if gt is not None:
            sub = gt[gt["source1_entity_id"].isin(set(blocked))]
            fnd, tot, missed, ranks = recall_against(blocked, sub)
            found += fnd
            total += tot
            ranks_all.update(ranks)
            missed_all.extend(missed[:10])
            if tot:
                print(f"  {country} recall: {fnd / tot:.4f} ({fnd:,}/{tot:,})", flush=True)
        del s1_c, cand_c, blocked
        gc.collect()

    for eid in s1["entity_id"].tolist():
        all_rows.setdefault(eid, [])

    n_pairs = sum(len(v) for v in all_rows.values())
    possible = len(s1) * max(len(cand), 1)
    print(
        f"\nPairs={n_pairs:,}  {candidate_stats(all_rows, args.top_k)}  "
        f"reduction={1 - n_pairs / possible:.6%}  time={time.time() - t0:.0f}s",
        flush=True,
    )
    if total:
        print(f"BLOCKING RECALL: {found / total:.4f} ({found:,}/{total:,})", flush=True)
        print(f"  rank of found true matches: {dict(sorted(ranks_all.items()))}", flush=True)
        for s1_id, mid in missed_all[:8]:
            print(f"  missed {s1_id} -> {mid}", flush=True)

    if args.mode == "test":
        name = "candidate_pairs.tsv"
    elif args.mode == "full":
        name = "train_candidate_pairs.tsv"
    else:
        name = "sample_candidate_pairs.tsv"
    path = os.path.join(out_dir, name)
    write_tsv(path, all_rows, s1["entity_id"].tolist())
    print(f"Wrote {path}", flush=True)


def self_test() -> None:
    """Reorder, typo, and address-only pairs must all be retrieved."""
    s1 = pd.DataFrame(
        [
            {
                "entity_id": "S1-a",
                "country": "US",
                "name_core": "alpha beta",
                "addr_normalized": "100 oak street austin tx",
                "addr_numbers": "100",
            },
            {
                "entity_id": "S1-b",
                "country": "US",
                "name_core": "westchester market",
                "addr_normalized": "551 pine road dallas tx",
                "addr_numbers": "551",
            },
            {
                "entity_id": "S1-c",
                "country": "US",
                "name_core": "totally different",
                "addr_normalized": "900 lakeside drive miami fl",
                "addr_numbers": "900",
            },
            {
                "entity_id": "S1-d",
                "country": "India",
                "name_core": "raj investments",
                "addr_normalized": "6 29 colony main road mylapore chennai",
                "addr_numbers": "6 29 2",
            },
        ]
    )
    cand = pd.DataFrame(
        [
            {
                "entity_id": "S2-a",
                "country": "US",
                "name_core": "beta alpha",
                "addr_normalized": "100 oak street austin tx",
                "addr_numbers": "100",
            },
            {
                "entity_id": "S2-b",
                "country": "US",
                "name_core": "westcheter market",
                "addr_normalized": "551 pine road dallas tx",
                "addr_numbers": "551",
            },
            {
                "entity_id": "S3-c",
                "country": "US",
                "name_core": "unrelated trading",
                "addr_normalized": "900 lakeside drive miami fl",
                "addr_numbers": "900",
            },
            {
                "entity_id": "S2-z",
                "country": "US",
                "name_core": "other shop",
                "addr_normalized": "1 main street boston ma",
                "addr_numbers": "1",
            },
            {
                "entity_id": "S2-d",
                "country": "India",
                "name_core": "raj invesdhmendhs elelbhi",
                "addr_normalized": "6 29 colony main road mylapore chennai",
                "addr_numbers": "6 29 2",
            },
        ]
    )
    got = block_frames(s1, cand, top_k=5, min_score=1.0, chunk=3)
    assert "S2-a" in got["S1-a"], f"word-reorder miss: {got}"
    assert "S2-b" in got["S1-b"], f"typo+shared-word miss: {got}"
    assert "S3-c" in got["S1-c"], f"addr-only miss: {got}"
    assert "S2-d" in got["S1-d"], f"addr-number miss: {got}"
    assert got["S1-a"][0] == "S2-a", f"true match should rank first: {got}"
    print("self-test OK", got)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blocking / candidate generation")
    p.add_argument("--mode", choices=["sample", "full", "test", "self-test"], default="sample")
    p.add_argument("--sample-size", type=int, default=100000)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--min-score", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=2000, help="Source-1 rows scored per sparse product")
    p.add_argument("--country", default="", help="Run a single country label, e.g. US")
    p.add_argument("--preprocessed", default="", help="Folder with *_s1.parquet files")
    p.add_argument("--output", default="", help="Output directory")
    args = p.parse_args()
    args.country = args.country.strip()
    args.preprocessed = args.preprocessed.strip() or None
    args.output = args.output.strip() or None
    return args


if __name__ == "__main__":
    cli = parse_args()
    if cli.mode == "self-test":
        self_test()
    else:
        run(cli)
