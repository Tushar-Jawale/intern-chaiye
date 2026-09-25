"""
Phase 3 — Pair features for the matcher.

Computed only on blocking candidates, never on the full cross product.
String metrics that mostly repeat each other are not all computed.
Two name signals and two address signals cover the rest:

  token Jaccard          shared words, ignores order
  token-sort ratio       typos inside the same words
  length ratio           truncation
  number Jaccard         same premises
  number-set equal       the 6/29/2 style match
  name-in-address        a name word sitting in the other address

rapidfuzz is optional. Without it, token-sort ratio falls back to an exact
sorted-token score.
"""
from __future__ import annotations

try:
    from rapidfuzz.fuzz import token_sort_ratio as _token_sort_ratio
except ImportError:
    _token_sort_ratio = None

FEATURE_COLS = [
    "name_jaccard",
    "name_sort",
    "name_len_ratio",
    "name_contains",
    "addr_jaccard",
    "addr_sort",
    "addr_len_ratio",
    "num_jaccard",
    "num_set_equal",
    "name_in_address",
]


def _tokens(text: str, min_len: int = 2) -> set[str]:
    if not text:
        return set()
    return {t for t in str(text).split() if len(t) >= min_len}


def _numbers(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t in str(text).split() if t.isdigit() and t != "0"}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _len_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    short, long = len(a), len(b)
    if short > long:
        short, long = long, short
    return short / long


def _sort_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _token_sort_ratio is not None:
        return _token_sort_ratio(a, b) / 100.0
    ta = " ".join(sorted(_tokens(a)))
    tb = " ".join(sorted(_tokens(b)))
    if not ta or not tb:
        return 0.0
    return 1.0 if ta == tb else 0.0


def _contains(a: str, b: str) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    return 1.0 if a in b or b in a else 0.0


def pair_features(name1: str, addr1: str, nums1: str, name2: str, addr2: str, nums2: str) -> list[float]:
    nt1, nt2 = _tokens(name1), _tokens(name2)
    at1, at2 = _tokens(addr1, 3), _tokens(addr2, 3)
    nu1, nu2 = _numbers(nums1), _numbers(nums2)
    name_in_addr = 0.0
    if nt1 and at2 and (nt1 & at2):
        name_in_addr = 1.0
    elif nt2 and at1 and (nt2 & at1):
        name_in_addr = 1.0
    return [
        _jaccard(nt1, nt2),
        _sort_ratio(name1, name2),
        _len_ratio(name1, name2),
        _contains(name1, name2),
        _jaccard(at1, at2),
        _sort_ratio(addr1, addr2),
        _len_ratio(addr1, addr2),
        _jaccard(nu1, nu2),
        1.0 if nu1 and nu1 == nu2 else 0.0,
        name_in_addr,
    ]


def macro_f05(preds: dict[str, set[str]], gold: dict[str, set[str]]) -> float:
    """Macro F0.5 over every entity in gold. Empty gold and empty pred scores 1."""
    if not gold:
        return 0.0
    total = 0.0
    for eid, truth in gold.items():
        pred = preds.get(eid) or set()
        if not truth and not pred:
            total += 1.0
            continue
        if not truth or not pred:
            continue
        hit = len(truth & pred)
        precision = hit / len(pred)
        recall = hit / len(truth)
        denom = 0.25 * precision + recall
        if denom:
            total += (1.25 * precision * recall) / denom
    return total / len(gold)


def self_test() -> None:
    same = pair_features(
        "raj investments", "6 29 colony mylapore", "6 29 2",
        "raj investments", "6 29 colony mylapore", "6 29 2",
    )
    assert same[0] == 1.0 and same[7] == 1.0 and same[8] == 1.0, same
    typo_addr = pair_features(
        "raj investments", "6 29 colony mylapore chennai", "6 29 2",
        "raj invesdhmendhs", "6 29 colony mylapore chennai", "6 29 2",
    )
    assert typo_addr[7] == 1.0 and typo_addr[8] == 1.0, typo_addr
    different = pair_features(
        "alpha beta", "100 oak street", "100",
        "other shop", "1 main street", "1",
    )
    assert different[0] == 0.0 and different[8] == 0.0, different

    gold = {
        "S1-1": {"S2-a", "S3-b"},
        "S1-2": set(),
        "S1-3": {"S2-c"},
    }
    # example from the problem statement: 2 of 3 predicted, both true ones found
    preds = {
        "S1-1": {"S2-a", "S2-x", "S3-b"},
        "S1-2": set(),
        "S1-3": set(),
    }
    score = macro_f05(preds, gold)
    # entity1 ~0.714, entity2 1.0, entity3 0.0
    assert abs(score - (0.7142857142857143 + 1.0) / 3) < 1e-6, score
    print("feature self-test OK", round(score, 4))


if __name__ == "__main__":
    self_test()
