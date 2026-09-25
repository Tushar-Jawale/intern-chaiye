"""Shared helpers for the v3 pipeline: loading normalized tables, labels, splits."""
from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NORM_COLS = ["entity_id", "country", "nm", "core", "phon", "sfx", "dom", "scr", "ad", "st", "nums", "units", "pin"]


def norm_path(work: str, stem: str) -> str:
    return os.path.join(work, "norm", stem + ".parquet")


def safe(country: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(country))


def countries(work: str, split: str) -> list[str]:
    t = pq.read_table(norm_path(work, f"{split}_s1"), columns=["country"]).to_pandas()
    return sorted(t["country"].dropna().unique().tolist())


def load_norm(work: str, stem: str, country: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or NORM_COLS
    filters = [("country", "=", country)] if country is not None else None
    df = pq.read_table(norm_path(work, stem), columns=cols, filters=filters).to_pandas()
    for c in cols:
        if df[c].dtype == object:
            df[c] = df[c].fillna("")
    return df.reset_index(drop=True)


def load_queries(work: str, split: str, country: str, columns: list[str] | None = None) -> pd.DataFrame:
    a = load_norm(work, f"{split}_s2", country, columns)
    b = load_norm(work, f"{split}_s3", country, columns)
    return pd.concat([a, b], ignore_index=True)


def bucket(ids, mod: int = 1000) -> np.ndarray:
    """Stable hash bucket per id (crc32), independent of Python's hash seed."""
    return np.fromiter((zlib.crc32(s.encode()) % mod for s in ids), dtype=np.int32, count=len(ids))


def owner_map(work: str, want: set[str] | None = None) -> dict[str, str]:
    """candidate id -> Source-1 id from train ground truth."""
    gt = pq.read_table(norm_path(work, "train_gt")).to_pandas()
    out: dict[str, str] = {}
    for s1, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not isinstance(matched, str) or not matched:
            continue
        for m in matched.split(","):
            if m and (want is None or m in want):
                out[m] = s1
    return out


def dev_filter(s1: pd.DataFrame, q: pd.DataFrame, owner: dict[str, str], frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A smaller world: a hash slice of Source-1, the records they own, and the
    same slice of unowned records. Records owned by dropped entities disappear."""
    if frac >= 1.0:
        return s1, q
    cut = int(frac * 1000)
    keep_s1 = bucket(s1["entity_id"].tolist()) < cut
    s1 = s1[keep_s1].reset_index(drop=True)
    kept = set(s1["entity_id"])
    own = [owner.get(e) for e in q["entity_id"]]
    qb = bucket(q["entity_id"].tolist()) < cut
    keep_q = np.fromiter(((o in kept) if o is not None else bool(b) for o, b in zip(own, qb)), dtype=bool, count=len(q))
    return s1, q[keep_q].reset_index(drop=True)


def ret_path(work: str, split: str, country: str) -> str:
    return os.path.join(work, "ret", f"{split}_{safe(country)}.npz")


def load_ret(work: str, split: str, country: str) -> dict:
    z = np.load(ret_path(work, split, country), allow_pickle=False)
    return {k: z[k] for k in z.files}
