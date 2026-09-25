"""
Decisions and the metric.

1. Each Source 2/3 record keeps only its best Source-1 candidate (a record has
   at most one owner).
2. Per Source-1 entity, either
     thr: keep records whose probability clears a threshold, or
     ef:  keep the top-k records that maximise expected F0.5 under calibrated,
          independent probabilities (k = 0 is scored as P(no true match)).
Both are tuned on the tuning entities; the leaderboard metric is macro F0.5
over every Source-1 entity, singletons included.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def macro_report(pred: dict[str, set[str]], gold: dict[str, set[str]]) -> dict:
    f_sum = p_sum = r_sum = 0.0
    groups: dict[str, list[float]] = {}
    sfp = sn = 0
    for eid, truth in gold.items():
        got = pred.get(eid) or set()
        if not truth:
            sn += 1
            f = 1.0 if not got else 0.0
            sfp += 0 if not got else 1
            p = r = f
        elif not got:
            f = p = r = 0.0
        else:
            hit = len(truth & got)
            p = hit / len(got)
            r = hit / len(truth)
            f = 1.25 * p * r / (0.25 * p + r) if hit else 0.0
        f_sum += f
        p_sum += p
        r_sum += r
        n = len(truth)
        key = "matches=0" if n == 0 else "matches=1" if n == 1 else "matches=2-3" if n <= 3 else "matches=4+"
        groups.setdefault(key, []).append(f)
    n = max(len(gold), 1)
    return {
        "f05": f_sum / n, "precision": p_sum / n, "recall": r_sum / n,
        "singleton_fp": sfp / max(sn, 1), "n": len(gold),
        "groups": {k: (float(np.mean(v)), len(v)) for k, v in sorted(groups.items())},
    }


def best_per_query(q: np.ndarray, s: np.ndarray, p: np.ndarray):
    """Arrays of pairs -> one (q, s, p) per query, the highest p."""
    if len(q) == 0:
        return q, s, p
    order = np.lexsort((-p, q))
    q, s, p = q[order], s[order], p[order]
    first = np.ones(len(q), dtype=bool)
    first[1:] = q[1:] != q[:-1]
    return q[first], s[first], p[first]


@njit(cache=False)
def _pb(p):
    """Poisson-binomial distribution of the number of successes."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for i in range(len(p)):
        for j in range(i + 1, 0, -1):
            d[j] = d[j] * (1.0 - p[i]) + d[j - 1] * p[i]
        d[0] *= 1.0 - p[i]
    return d


@njit(cache=False)
def _best_k(p, extra):
    """p sorted descending. Returns k maximising expected F0.5."""
    n = len(p)
    best_k = 0
    none = 1.0
    for i in range(n):
        none *= 1.0 - p[i]
    best_v = none if extra <= 0 else none * np.exp(-extra)
    for k in range(1, n + 1):
        sel = _pb(p[:k])
        uns = _pb(p[k:])
        v = 0.0
        for a in range(1, k + 1):
            if sel[a] == 0.0:
                continue
            for u in range(0, n - k + 1):
                if uns[u] == 0.0:
                    continue
                v += sel[a] * uns[u] * 1.25 * a / (0.25 * (a + u + extra) + k)
        if v > best_v:
            best_v = v
            best_k = k
    return best_k


@njit(cache=False)
def _ef_select(ent_start, ent_len, probs, extra):
    keep = np.zeros(len(probs), dtype=np.bool_)
    for e in range(len(ent_start)):
        a = ent_start[e]
        n = ent_len[e]
        k = _best_k(probs[a:a + n], extra)
        for i in range(k):
            keep[a + i] = True
    return keep


def select(bq: np.ndarray, bs: np.ndarray, bp: np.ndarray, method: str, thr: float = 0.5,
           floor: float = 0.05, extra: float = 0.0, calib=None) -> np.ndarray:
    """Given one best pair per query, return a mask of pairs to output."""
    if len(bq) == 0:
        return np.zeros(0, dtype=bool)
    if method == "thr":
        return bp >= thr
    p = calib(bp) if calib is not None else bp
    cand = np.flatnonzero(bp >= floor)
    order = cand[np.lexsort((-p[cand], bs[cand]))]
    s_sorted = bs[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = s_sorted[1:] != s_sorted[:-1]
    starts = np.flatnonzero(first)
    lens = np.diff(np.append(starts, len(order)))
    keep_sorted = _ef_select(starts.astype(np.int64), lens.astype(np.int64), p[order].astype(np.float64), float(extra))
    mask = np.zeros(len(bq), dtype=bool)
    mask[order[keep_sorted]] = True
    return mask


def to_pred(bq, bs, mask, q_ids, s_ids) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for qi, si in zip(bq[mask].tolist(), bs[mask].tolist()):
        out.setdefault(s_ids[si], set()).add(q_ids[qi])
    return out


def self_test() -> None:
    gold = {"S1-1": {"a", "b"}, "S1-2": set(), "S1-3": {"c"}}
    rep = macro_report({"S1-1": {"a", "b", "x"}, "S1-2": set(), "S1-3": set()}, gold)
    assert abs(rep["f05"] - (0.7142857142857143 + 1.0) / 3) < 1e-9, rep
    assert _best_k(np.array([0.9, 0.2]), 0.0) == 1
    assert _best_k(np.array([0.3]), 0.0) == 0
    assert _best_k(np.array([0.95, 0.9, 0.8]), 0.0) == 3
    bq = np.array([0, 1, 2]); bs = np.array([0, 0, 1]); bp = np.array([0.9, 0.4, 0.7])
    m = select(bq, bs, bp, "ef", floor=0.05)
    assert m.tolist() == [True, False, True], m
    print("decide3 self-test OK", flush=True)


if __name__ == "__main__":
    self_test()
