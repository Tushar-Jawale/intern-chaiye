"""
Phase 2 — Blocking / candidate generation.

Measured on 4,000 true train pairs:
  same country                         100%
  shared name word or address number    90.5%
  typo-only (character n-grams)          4.8%

Each country is blocked with inverted indexes on name words, name-word pairs,
address words, address-word pairs, and address numbers. Character n-grams are
not built. On the 20k sample almost every row already fills the top-200 list,
so a second pass on empty rows does not reach the pairs that were missed.

A key is stored only when it is shared by at most max_df candidates, so common
words never become a giant posting list. Each Source-1 row keeps the top-K
candidates by IDF score. That list is the matcher input and the
candidate_pairs.tsv row.

No extra packages. In a Kaggle notebook, paste this file into one cell, delete
the ``if __name__ == "__main__"`` block at the bottom, run the cell, then:

    from argparse import Namespace
    import glob, os
    hits = glob.glob("/kaggle/input/**/train_s1.parquet", recursive=True)
    args = Namespace(
        mode="full", sample_size=20000, top_k=200, min_score=1.0,
        country="", preprocessed=os.path.dirname(hits[0]),
        output="/kaggle/working/output",
    )
    run(args)
"""
from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import time
from collections import Counter, defaultdict

import pandas as pd

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


class _Index:
    def __init__(self, n_cands: int, max_df: int, base: float):
        self.n = n_cands
        self.max_df = max_df
        self.base = base
        self.counts: Counter = Counter()
        self.postings: dict[str, list[int]] = {}
        self.weight: dict[str, float] = {}

    def observe(self, keys: list[str]) -> None:
        if keys:
            self.counts.update(set(keys))

    def freeze(self) -> None:
        keep = {k for k, c in self.counts.items() if 1 <= c <= self.max_df}
        self.postings = {k: [] for k in keep}
        self.weight = {k: self.base * _idf(self.counts[k], self.n) for k in keep}
        self._keep = keep

    def add(self, i: int, keys: list[str]) -> None:
        keep = self._keep
        postings = self.postings
        for k in set(keys):
            if k in keep:
                postings[k].append(i)

    def release_counts(self) -> None:
        self.counts.clear()


def _hit(scores: dict[int, float], index: _Index, keys: list[str]) -> None:
    weight = index.weight
    postings = index.postings
    for k in set(keys):
        w = weight.get(k)
        if w is None:
            continue
        for cid in postings[k]:
            scores[cid] = scores.get(cid, 0.0) + w


def block_frames(
    s1: pd.DataFrame,
    cand: pd.DataFrame,
    top_k: int = 200,
    min_score: float = 1.0,
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

    name_keys: list[list[str]] = []
    name_pair_keys: list[list[str]] = []
    addr_keys: list[list[str]] = []
    addr_pair_keys: list[list[str]] = []
    num_keys: list[list[str]] = []
    sig_keys: list[list[str]] = []
    comp_keys: list[list[str]] = []
    pref_keys: list[list[str]] = []
    exact_keys: list[str] = []
    cand_sig: list[str] = []
    cand_long: list[set[str]] = []

    for i in range(n):
        words = _words(c_name[i], 3)
        addr_words = _words(c_addr[i], 3)
        addr_pair_src = [w for w in addr_words if len(w) >= 4]
        nums = _all_numbers(c_nums[i])
        name_keys.append(words)
        name_pair_keys.append(_pairs(words))
        addr_keys.append(addr_words)
        addr_pair_keys.append(_pairs(addr_pair_src))
        num_keys.append([t for t in nums if len(t) >= 3])
        sig = _signature(nums)
        sig_keys.append(sig)
        comp_keys.append(_composites(nums, addr_words))
        pref = _prefix(words)
        pref_keys.append([pref] if pref else [])
        exact_keys.append(" ".join(sorted(set(words))) if words else "")
        cand_sig.append(sig[0] if sig else "")
        cand_long.append({t for t in nums if len(t) >= 3})

    # Single-word caps stay under top_k. Common words are not a blocking key
    # by themselves; they survive inside a pair, which is rare.
    specs = {
        "name": _Index(n, 1500, 1.0),
        "npair": _Index(n, 8000, 2.4),
        "addr": _Index(n, 1500, 0.9),
        "apair": _Index(n, 4000, 2.0),
        "num": _Index(n, 800, 1.2),
        "sig": _Index(n, 200, 2.8),
        "comp": _Index(n, 500, 2.2),
        "pref": _Index(n, 300, 0.8),
    }
    rows = {
        "name": name_keys,
        "npair": name_pair_keys,
        "addr": addr_keys,
        "apair": addr_pair_keys,
        "num": num_keys,
        "sig": sig_keys,
        "comp": comp_keys,
        "pref": pref_keys,
    }
    print(f"      counting keys on {n:,} candidates", flush=True)
    for kind, index in specs.items():
        for keys in rows[kind]:
            index.observe(keys)
        index.freeze()
    print("      filling indexes", flush=True)
    for kind, index in specs.items():
        for i, keys in enumerate(rows[kind]):
            index.add(i, keys)
        index.release_counts()

    exact_counts = Counter(k for k in exact_keys if k)
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(exact_keys):
        if key and exact_counts[key] <= 400:
            exact_groups[key].append(i)

    del name_keys, name_pair_keys, addr_keys, addr_pair_keys
    del num_keys, sig_keys, comp_keys, pref_keys, exact_keys
    del c_name, c_addr, c_nums
    gc.collect()

    out: dict[str, list[str]] = {}
    t0 = time.time()
    report_every = 100_000
    for i, eid in enumerate(s1_ids):
        scores: dict[int, float] = {}
        words = _words(s1_name[i], 3)
        addr_words = _words(s1_addr[i], 3)
        nums = _all_numbers(s1_nums[i])
        _hit(scores, specs["name"], words)
        _hit(scores, specs["npair"], _pairs(words))
        _hit(scores, specs["addr"], addr_words)
        _hit(scores, specs["apair"], _pairs([w for w in addr_words if len(w) >= 4]))
        _hit(scores, specs["num"], [t for t in nums if len(t) >= 3])
        _hit(scores, specs["sig"], _signature(nums))
        _hit(scores, specs["comp"], _composites(nums, addr_words))
        pref = _prefix(words)
        if pref:
            _hit(scores, specs["pref"], [pref])
        exact = " ".join(sorted(set(words))) if words else ""
        group = exact_groups.get(exact, ())
        if group:
            qsig = _signature(nums)
            qsig = qsig[0] if qsig else ""
            qlong = {t for t in nums if len(t) >= 3}
            df = exact_counts[exact]
            for cid in group:
                if df <= 80 or (qsig and cand_sig[cid] == qsig) or (qlong and qlong & cand_long[cid]):
                    scores[cid] = scores.get(cid, 0.0) + 40.0

        if not scores:
            out[eid] = []
        else:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out[eid] = [c_ids[cid] for cid, sc in ranked[:top_k] if sc >= min_score]

        if (i + 1) % report_every == 0 or i + 1 == len(s1_ids):
            print(
                f"      scored {i + 1:,}/{len(s1_ids):,} in {time.time() - t0:.0f}s",
                flush=True,
            )

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


def recall_against(candidates: dict[str, list[str]], gt: pd.DataFrame) -> tuple[int, int, list]:
    found = 0
    total = 0
    missed = []
    id_set = {k: set(v) for k, v in candidates.items()}
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if s1_id not in id_set:
            continue
        if matched is None or (isinstance(matched, float) and pd.isna(matched)):
            continue
        text = str(matched).strip()
        if not text or text == "nan":
            continue
        got = id_set[s1_id]
        for mid in text.split(","):
            mid = mid.strip()
            if not mid:
                continue
            total += 1
            if mid in got:
                found += 1
            elif len(missed) < 15:
                missed.append((s1_id, mid))
    return found, total, missed


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
        blocked = block_frames(s1_c, cand_c, top_k=args.top_k, min_score=args.min_score)
        all_rows.update(blocked)
        n_pairs = sum(len(v) for v in blocked.values())
        print(
            f"  {country} done in {time.time() - t_c:.0f}s, pairs={n_pairs:,}, "
            f"avg={n_pairs / max(len(blocked), 1):.1f}",
            flush=True,
        )
        if gt is not None:
            sub = gt[gt["source1_entity_id"].isin(set(blocked))]
            fnd, tot, missed = recall_against(blocked, sub)
            found += fnd
            total += tot
            missed_all.extend(missed[:10])
            if tot:
                print(f"  recall so far: {found / total:.4f} ({found:,}/{total:,})", flush=True)
        del s1_c, cand_c, blocked
        gc.collect()

    for eid in s1["entity_id"].tolist():
        all_rows.setdefault(eid, [])

    n_pairs = sum(len(v) for v in all_rows.values())
    n_with = sum(1 for v in all_rows.values() if v)
    possible = len(s1) * max(len(cand), 1)
    print(
        f"\nPairs={n_pairs:,}  with_candidates={n_with:,}/{len(all_rows):,}  "
        f"avg={n_pairs / max(len(all_rows), 1):.1f}  "
        f"reduction={1 - n_pairs / possible:.6%}  time={time.time() - t0:.0f}s",
        flush=True,
    )
    if total:
        print(f"BLOCKING RECALL: {found / total:.4f} ({found:,}/{total:,})", flush=True)
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
                "name_normalized": "alpha beta llc",
                "addr_normalized": "100 oak street austin tx",
                "addr_numbers": "100",
            },
            {
                "entity_id": "S1-b",
                "country": "US",
                "name_core": "westchester market",
                "name_normalized": "westchester market",
                "addr_normalized": "551 pine road dallas tx",
                "addr_numbers": "551",
            },
            {
                "entity_id": "S1-c",
                "country": "US",
                "name_core": "totally different",
                "name_normalized": "totally different",
                "addr_normalized": "900 lakeside drive miami fl",
                "addr_numbers": "900",
            },
            {
                "entity_id": "S1-d",
                "country": "India",
                "name_core": "raj investments",
                "name_normalized": "raj investments",
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
                "name_normalized": "beta alpha",
                "addr_normalized": "100 oak street austin tx",
                "addr_numbers": "100",
            },
            {
                "entity_id": "S2-b",
                "country": "US",
                "name_core": "westcheter market",
                "name_normalized": "westcheter market",
                "addr_normalized": "551 pine road dallas tx",
                "addr_numbers": "551",
            },
            {
                "entity_id": "S3-c",
                "country": "US",
                "name_core": "unrelated trading",
                "name_normalized": "unrelated trading",
                "addr_normalized": "900 lakeside drive miami fl",
                "addr_numbers": "900",
            },
            {
                "entity_id": "S2-z",
                "country": "US",
                "name_core": "other shop",
                "name_normalized": "other shop",
                "addr_normalized": "1 main street boston ma",
                "addr_numbers": "1",
            },
            {
                "entity_id": "S2-d",
                "country": "India",
                "name_core": "raj invesdhmendhs elelbhi",
                "name_normalized": "raj invesdhmendhs elelbhi",
                "addr_normalized": "6 29 colony main road mylapore chennai",
                "addr_numbers": "6 29 2",
            },
        ]
    )
    got = block_frames(s1, cand, top_k=5, min_score=1.0)
    assert "S2-a" in got["S1-a"], f"word-reorder miss: {got}"
    assert "S2-b" in got["S1-b"], f"typo+shared-word miss: {got}"
    assert "S3-c" in got["S1-c"], f"addr-only miss: {got}"
    assert "S2-d" in got["S1-d"], f"addr-number miss: {got}"
    assert got["S1-a"][0] == "S2-a", f"true match should rank first: {got}"
    print("self-test OK", got)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blocking / candidate generation")
    p.add_argument("--mode", choices=["sample", "full", "test", "self-test"], default="sample")
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--min-score", type=float, default=1.0)
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
