"""
Pair features v3.

Records of one country are encoded once into integer arrays (word ids, numbers,
unit ids, state, legal-form bits) and string arrays. Pair features then come
from a numba kernel (set overlaps with IDF weights, number agreement including
one-digit typos, unit/state/PIN/legal-form agreement) and rapidfuzz cpdist
(C++ string similarities). Nothing loops over pairs in Python.

IDF is computed on the Source-1 records of the same split and country and
divided by log(N+1), so a word's weight means the same thing for a 1.3M-row
train index and a 260k-row test index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

W_NAME, W_ADDR, W_PH, W_NUM, W_UNIT = 8, 16, 8, 8, 4

LEGAL_BITS = {name: 1 << i for i, name in enumerate([
    "private", "limited", "inc", "corp", "co", "llc", "llp", "lp", "pllc", "plc", "pc", "pa", "opc",
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scop", "ei", "eirl", "scp", "scs", "selarl", "gie", "ets",
])}

KERNEL_COLS = [
    "n_wjacc", "n_wcov_min", "n_wcov_q", "n_wcov_s", "n_shared", "n_maxw_shared", "n_maxw_unq_q",
    "n_maxw_unq_s", "n_first_eq", "n_cnt_q", "n_cnt_s", "ph_jacc",
    "a_wjacc", "a_wcov_min", "a_maxw_shared", "a_cnt_q", "a_cnt_s",
    "num_jacc", "num_shared", "num_maxlen_shared", "num_near", "num_conflict", "num_cnt_q", "num_cnt_s",
    "unit_shared", "unit_conflict", "pin_eq", "pin_conflict", "st_eq", "st_conflict",
    "sfx_eq", "sfx_conflict", "name_in_addr", "n_unseen_q",
]
STRING_COLS = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "ph_tset", "n_acr",
    "a_tset", "a_tsort", "a_partial", "nums_tset", "lnum_sim",
]
FLAG_COLS = ["dom_q", "scr_q", "s3_q", "q_addr_empty", "s_addr_empty", "q_name_len", "s_name_len", "q_name_empty"]
RET_COLS = ["ret_rank", "ret_score", "ret_rel", "ret_gap_q", "ret_ncand", "ret_next_gap"]
FEATURES = KERNEL_COLS + STRING_COLS + FLAG_COLS + RET_COLS


class Table:
    """Encoded records of one country (Source 1 or queries)."""

    def __init__(self, n: int):
        self.n = n


def _explode(series: pd.Series, width: int) -> pd.DataFrame:
    tok = series.str.split().explode()
    tok = tok[tok.notna() & (tok != "")]
    frame = pd.DataFrame({"r": tok.index.to_numpy(np.int64), "t": tok.to_numpy()})
    frame = frame.drop_duplicates()
    frame["p"] = frame.groupby("r").cumcount().to_numpy()
    return frame[frame["p"] < width]


def _fill(n: int, width: int, frame: pd.DataFrame, values: np.ndarray, dtype, pad) -> np.ndarray:
    out = np.full((n, width), pad, dtype=dtype)
    out[frame["r"].to_numpy(), frame["p"].to_numpy()] = values
    return out


def _numbers(series: pd.Series, n: int):
    frame = _explode(series, W_NUM)
    txt = frame["t"].astype(str)
    vals = txt.str[-18:].astype(np.int64).to_numpy()
    lens = txt.str.len().clip(upper=18).to_numpy(np.int8)
    return _fill(n, W_NUM, frame, vals, np.int64, -1), _fill(n, W_NUM, frame, lens, np.int8, 0)


def _acronyms(core: list[str]) -> tuple[list[str], list[str], list[str]]:
    a1, a2, aa = [], [], []
    for c in core:
        t = c.split()
        if len(t) >= 2:
            a1.append(t[0][0] + "".join(t[1:]))
            a2.append(t[0][0] + t[1][0] + "".join(t[2:]))
            aa.append("".join(w[0] for w in t))
        else:
            a1.append("")
            a2.append("")
            aa.append("")
    return a1, a2, aa


def build_tables(s1: pd.DataFrame, q: pd.DataFrame) -> tuple[Table, Table, dict]:
    """Encode Source-1 and query records of one country with a shared vocabulary."""
    n1, nq = len(s1), len(q)
    frames = {}
    for tag, df in (("s", s1), ("q", q)):
        frames[tag + "n"] = _explode(df["core"], W_NAME)
        frames[tag + "a"] = _explode(df["ad"].str.replace(r"\b\d+\b", " ", regex=True), W_ADDR)
        frames[tag + "p"] = _explode(df["phon"], W_PH)
        frames[tag + "u"] = _explode(df["units"], W_UNIT)
    word_tok = pd.concat([frames["sn"]["t"], frames["sa"]["t"], frames["qn"]["t"], frames["qa"]["t"]], ignore_index=True)
    word_ids, word_uni = pd.factorize(word_tok)
    ph_ids, _ = pd.factorize(pd.concat([frames["sp"]["t"], frames["qp"]["t"]], ignore_index=True))
    unit_ids, _ = pd.factorize(pd.concat([frames["su"]["t"], frames["qu"]["t"]], ignore_index=True))
    st_ids, _ = pd.factorize(pd.concat([s1["st"], q["st"]], ignore_index=True).replace("", np.nan))
    n_words = len(word_uni)
    sizes = [len(frames[k]) for k in ("sn", "sa", "qn", "qa")]
    cut = np.cumsum([0] + sizes)
    ids = {k: word_ids[cut[i]:cut[i + 1]] for i, k in enumerate(("sn", "sa", "qn", "qa"))}
    denom = np.log(n1 + 1.0)
    df_name = np.bincount(ids["sn"], minlength=n_words).astype(np.float64)
    df_addr = np.bincount(ids["sa"], minlength=n_words).astype(np.float64)
    w_name = (np.log((n1 + 1.0) / (df_name + 1.0)) / denom).astype(np.float32)
    w_addr = (np.log((n1 + 1.0) / (df_addr + 1.0)) / denom).astype(np.float32)
    ph_cut = len(frames["sp"])
    u_cut = len(frames["su"])
    tables = []
    for tag, df, n in (("s", s1, n1), ("q", q, nq)):
        t = Table(n)
        t.name = _fill(n, W_NAME, frames[tag + "n"], ids[tag + "n"].astype(np.int32), np.int32, -1)
        t.addr = _fill(n, W_ADDR, frames[tag + "a"], ids[tag + "a"].astype(np.int32), np.int32, -1)
        pid = ph_ids[:ph_cut] if tag == "s" else ph_ids[ph_cut:]
        t.ph = _fill(n, W_PH, frames[tag + "p"], pid.astype(np.int32), np.int32, -1)
        uid = unit_ids[:u_cut] if tag == "s" else unit_ids[u_cut:]
        t.unit = _fill(n, W_UNIT, frames[tag + "u"], uid.astype(np.int32), np.int32, -1)
        t.num, t.numlen = _numbers(df["nums"], n)
        stv = st_ids[:n1] if tag == "s" else st_ids[n1:]
        t.st = (stv + 1).astype(np.int32)
        t.pin = pd.to_numeric(df["pin"].replace("", "0"), errors="coerce").fillna(0).astype(np.int64).to_numpy()
        bits = np.zeros(n, np.int64)
        sf = _explode(df["sfx"], 8)
        if len(sf):
            b = sf["t"].map(LEGAL_BITS).fillna(0).astype(np.int64).to_numpy()
            np.add.at(bits, sf["r"].to_numpy(), b)
        t.sfx = bits
        t.dom = df["dom"].to_numpy(np.int8) if "dom" in df else np.zeros(n, np.int8)
        t.scr = df["scr"].to_numpy(np.int8) if "scr" in df else np.zeros(n, np.int8)
        t.s3 = df["entity_id"].str.startswith("S3-").to_numpy(np.int8)
        core = df["core"].tolist()
        t.core = np.array(core, dtype=object)
        t.comp = np.array([c.replace(" ", "") for c in core], dtype=object)
        t.ad = df["ad"].to_numpy(dtype=object)
        t.nums = df["nums"].to_numpy(dtype=object)
        t.phon = df["phon"].to_numpy(dtype=object)
        t.lnum = np.array([max(x.split(), key=len) if x else "" for x in df["nums"].tolist()], dtype=object)
        a1, a2, aa = _acronyms(core)
        t.acr1 = np.array(a1, dtype=object)
        t.acr2 = np.array(a2, dtype=object)
        t.acra = np.array(aa, dtype=object)
        t.core_len = np.array([len(c) for c in core], np.int16)
        t.ad_len = df["ad"].str.len().to_numpy(np.int32)
        t.nums_len = df["nums"].str.len().to_numpy(np.int32)
        t.phon_len = df["phon"].str.len().to_numpy(np.int32)
        tables.append(t)
    info = {"w_name": w_name, "w_addr": w_addr, "df_name": df_name.astype(np.int64)}
    return tables[0], tables[1], info


@njit
def _lev_le1(a, la, b, lb):
    if abs(la - lb) > 1:
        return False
    da = np.empty(19, np.int64)
    db = np.empty(19, np.int64)
    x = a
    for i in range(la - 1, -1, -1):
        da[i] = x % 10
        x //= 10
    x = b
    for i in range(lb - 1, -1, -1):
        db[i] = x % 10
        x //= 10
    i = 0
    j = 0
    edits = 0
    while i < la and j < lb:
        if da[i] == db[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if la > lb:
            i += 1
        elif lb > la:
            j += 1
        else:
            i += 1
            j += 1
    edits += (la - i) + (lb - j)
    return edits <= 1


@njit(parallel=True)
def _kernel(qi, si, QN, SN, QA, SA, QP, SP, QX, QXL, SX, SXL, QU, SU, Qpin, Spin, Qst, Sst, Qsfx, Ssfx,
            w_name, w_addr, df_name, out):
    n = qi.shape[0]
    for k in prange(n):
        a = qi[k]
        b = si[k]
        # ---- name words, IDF weighted
        tq = 0.0
        ts = 0.0
        sh = 0.0
        cnt = 0
        mxs = 0.0
        mxuq = 0.0
        mxus = 0.0
        nq = 0
        ns = 0
        unseen = 0
        for i in range(QN.shape[1]):
            x = QN[a, i]
            if x < 0:
                break
            nq += 1
            w = w_name[x]
            tq += w
            if df_name[x] == 0:
                unseen += 1
            found = False
            for j in range(SN.shape[1]):
                y = SN[b, j]
                if y < 0:
                    break
                if y == x:
                    found = True
                    break
            if found:
                sh += w
                cnt += 1
                if w > mxs:
                    mxs = w
            elif w > mxuq:
                mxuq = w
        for j in range(SN.shape[1]):
            y = SN[b, j]
            if y < 0:
                break
            ns += 1
            w = w_name[y]
            ts += w
            found = False
            for i in range(QN.shape[1]):
                x = QN[a, i]
                if x < 0:
                    break
                if x == y:
                    found = True
                    break
            if not found and w > mxus:
                mxus = w
        union = tq + ts - sh
        out[k, 0] = sh / union if union > 0 else 0.0
        mn = min(tq, ts)
        out[k, 1] = sh / mn if mn > 0 else 0.0
        out[k, 2] = sh / tq if tq > 0 else 0.0
        out[k, 3] = sh / ts if ts > 0 else 0.0
        out[k, 4] = cnt
        out[k, 5] = mxs
        out[k, 6] = mxuq
        out[k, 7] = mxus
        out[k, 8] = 1.0 if nq > 0 and ns > 0 and QN[a, 0] == SN[b, 0] else 0.0
        out[k, 9] = nq
        out[k, 10] = ns
        out[k, 33] = unseen
        # ---- phonetic skeletons
        pq_ = 0
        ps_ = 0
        pi_ = 0
        for i in range(QP.shape[1]):
            x = QP[a, i]
            if x < 0:
                break
            pq_ += 1
            for j in range(SP.shape[1]):
                y = SP[b, j]
                if y < 0:
                    break
                if y == x:
                    pi_ += 1
                    break
        for j in range(SP.shape[1]):
            if SP[b, j] < 0:
                break
            ps_ += 1
        pu = pq_ + ps_ - pi_
        out[k, 11] = pi_ / pu if pu > 0 else 0.0
        # ---- address words
        tq = 0.0
        ts = 0.0
        sh = 0.0
        mxs = 0.0
        nq_a = 0
        ns_a = 0
        for i in range(QA.shape[1]):
            x = QA[a, i]
            if x < 0:
                break
            nq_a += 1
            w = w_addr[x]
            tq += w
            for j in range(SA.shape[1]):
                y = SA[b, j]
                if y < 0:
                    break
                if y == x:
                    sh += w
                    if w > mxs:
                        mxs = w
                    break
        for j in range(SA.shape[1]):
            y = SA[b, j]
            if y < 0:
                break
            ns_a += 1
            ts += w_addr[y]
        union = tq + ts - sh
        out[k, 12] = sh / union if union > 0 else 0.0
        mn = min(tq, ts)
        out[k, 13] = sh / mn if mn > 0 else 0.0
        out[k, 14] = mxs
        out[k, 15] = nq_a
        out[k, 16] = ns_a
        # ---- numbers
        cq = 0
        cs = 0
        shared = 0
        maxlen = 0
        for i in range(QX.shape[1]):
            if QX[a, i] < 0:
                break
            cq += 1
        for j in range(SX.shape[1]):
            if SX[b, j] < 0:
                break
            cs += 1
        near = 0
        long_q = 0
        long_s = 0
        for i in range(cq):
            x = QX[a, i]
            lx = QXL[a, i]
            if lx >= 2:
                long_q += 1
            hit = False
            for j in range(cs):
                if SX[b, j] == x:
                    hit = True
                    break
            if hit:
                shared += 1
                if lx > maxlen:
                    maxlen = lx
            elif lx >= 2:
                for j in range(cs):
                    ly = SXL[b, j]
                    if ly >= 2 and _lev_le1(x, lx, SX[b, j], ly):
                        near += 1
                        break
        for j in range(cs):
            if SXL[b, j] >= 2:
                long_s += 1
        nu = cq + cs - shared
        out[k, 17] = shared / nu if nu > 0 else 0.0
        out[k, 18] = shared
        out[k, 19] = maxlen
        out[k, 20] = near
        out[k, 21] = 1.0 if long_q > 0 and long_s > 0 and shared == 0 and near == 0 else 0.0
        out[k, 22] = cq
        out[k, 23] = cs
        # ---- units, pin, state, legal form
        uq = 0
        us = 0
        ush = 0
        for i in range(QU.shape[1]):
            x = QU[a, i]
            if x < 0:
                break
            uq += 1
            for j in range(SU.shape[1]):
                y = SU[b, j]
                if y < 0:
                    break
                if y == x:
                    ush += 1
                    break
        for j in range(SU.shape[1]):
            if SU[b, j] < 0:
                break
            us += 1
        out[k, 24] = ush
        out[k, 25] = 1.0 if uq > 0 and us > 0 and ush == 0 else 0.0
        pa = Qpin[a]
        pb = Spin[b]
        out[k, 26] = 1.0 if pa > 0 and pa == pb else 0.0
        out[k, 27] = 1.0 if pa > 0 and pb > 0 and pa != pb else 0.0
        sa_ = Qst[a]
        sb_ = Sst[b]
        out[k, 28] = 1.0 if sa_ > 0 and sa_ == sb_ else 0.0
        out[k, 29] = 1.0 if sa_ > 0 and sb_ > 0 and sa_ != sb_ else 0.0
        fa = Qsfx[a]
        fb = Ssfx[b]
        out[k, 30] = 1.0 if fa != 0 and fa == fb else 0.0
        out[k, 31] = 1.0 if fa != 0 and fb != 0 and (fa & fb) == 0 else 0.0
        # ---- a name word sitting in the other address
        best = 0.0
        for i in range(QN.shape[1]):
            x = QN[a, i]
            if x < 0:
                break
            for j in range(SA.shape[1]):
                y = SA[b, j]
                if y < 0:
                    break
                if y == x and w_name[x] > best:
                    best = w_name[x]
        for j in range(SN.shape[1]):
            y = SN[b, j]
            if y < 0:
                break
            for i in range(QA.shape[1]):
                x = QA[a, i]
                if x < 0:
                    break
                if x == y and w_name[y] > best:
                    best = w_name[y]
        out[k, 32] = best


def kernel_features(TQ: Table, TS: Table, qa: np.ndarray, sa: np.ndarray, info: dict) -> np.ndarray:
    out = np.zeros((len(qa), len(KERNEL_COLS)), np.float32)
    if len(qa) == 0:
        return out
    _kernel(qa.astype(np.int64), sa.astype(np.int64), TQ.name, TS.name, TQ.addr, TS.addr, TQ.ph, TS.ph,
            TQ.num, TQ.numlen, TS.num, TS.numlen, TQ.unit, TS.unit, TQ.pin, TS.pin, TQ.st, TS.st,
            TQ.sfx, TS.sfx, info["w_name"], info["w_addr"], info["df_name"], out)
    return out


def _cp(scorer, a, b, workers: int) -> np.ndarray:
    from rapidfuzz import process

    return np.asarray(process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32), dtype=np.float32) / 100.0


def string_features(TQ: Table, TS: Table, qa: np.ndarray, sa: np.ndarray, workers: int = -1) -> np.ndarray:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein

    n = len(qa)
    out = np.zeros((n, len(STRING_COLS)), np.float32)
    if n == 0:
        return out
    cq, cs = TQ.core[qa], TS.core[sa]
    pq_, ps_ = TQ.comp[qa], TS.comp[sa]
    name_ok = (TQ.core_len[qa] > 0) & (TS.core_len[sa] > 0)
    out[:, 0] = _cp(fuzz.ratio, cq, cs, workers)
    out[:, 1] = _cp(fuzz.token_sort_ratio, cq, cs, workers)
    out[:, 2] = _cp(fuzz.token_set_ratio, cq, cs, workers)
    out[:, 3] = _cp(fuzz.partial_ratio, pq_, ps_, workers)
    out[:, 4] = _cp(JaroWinkler.normalized_similarity, pq_, ps_, workers) * 100.0
    ph_ok = (TQ.phon_len[qa] > 0) & (TS.phon_len[sa] > 0)
    out[:, 5] = _cp(fuzz.token_set_ratio, TQ.phon[qa], TS.phon[sa], workers)
    acr = np.maximum(_cp(fuzz.ratio, pq_, TS.acr1[sa], workers), _cp(fuzz.ratio, pq_, TS.acr2[sa], workers))
    acr = np.maximum(acr, _cp(fuzz.ratio, pq_, TS.acra[sa], workers))
    acr = np.maximum(acr, _cp(fuzz.ratio, ps_, TQ.acra[qa], workers))
    out[:, 6] = acr
    ad_ok = (TQ.ad_len[qa] > 0) & (TS.ad_len[sa] > 0)
    aq, as_ = TQ.ad[qa], TS.ad[sa]
    out[:, 7] = _cp(fuzz.token_set_ratio, aq, as_, workers)
    out[:, 8] = _cp(fuzz.token_sort_ratio, aq, as_, workers)
    out[:, 9] = _cp(fuzz.partial_ratio, aq, as_, workers)
    num_ok = (TQ.nums_len[qa] > 0) & (TS.nums_len[sa] > 0)
    out[:, 10] = _cp(fuzz.token_set_ratio, TQ.nums[qa], TS.nums[sa], workers)
    out[:, 11] = _cp(Levenshtein.normalized_similarity, TQ.lnum[qa], TS.lnum[sa], workers) * 100.0
    for cols, ok in (((0, 1, 2, 3, 4, 6), name_ok), ((5,), ph_ok), ((7, 8, 9), ad_ok), ((10, 11), num_ok)):
        for c in cols:
            out[~ok, c] = 0.0
    return out


def flag_features(TQ: Table, TS: Table, qa: np.ndarray, sa: np.ndarray) -> np.ndarray:
    out = np.zeros((len(qa), len(FLAG_COLS)), np.float32)
    if len(qa) == 0:
        return out
    out[:, 0] = TQ.dom[qa]
    out[:, 1] = TQ.scr[qa]
    out[:, 2] = TQ.s3[qa]
    out[:, 3] = TQ.ad_len[qa] == 0
    out[:, 4] = TS.ad_len[sa] == 0
    out[:, 5] = TQ.core_len[qa]
    out[:, 6] = TS.core_len[sa]
    out[:, 7] = TQ.core_len[qa] == 0
    return out


def retrieval_features(qa: np.ndarray, rank: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Pairs must be grouped by query and ordered by rank inside a group."""
    n = len(qa)
    out = np.zeros((n, len(RET_COLS)), np.float32)
    if n == 0:
        return out
    start = np.ones(n, dtype=bool)
    start[1:] = qa[1:] != qa[:-1]
    grp = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    best = score[first][grp]
    size = np.diff(np.append(first, n))[grp]
    second = np.zeros(len(first), np.float32)
    has2 = np.diff(np.append(first, n)) > 1
    second[has2] = score[first[has2] + 1]
    nxt = np.zeros(n, np.float32)
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = qa[1:] == qa[:-1]
    nxt[:-1][same_next[:-1]] = score[1:][same_next[:-1]]
    out[:, 0] = rank
    out[:, 1] = np.log1p(score)
    out[:, 2] = np.where(best > 0, score / np.maximum(best, 1e-6), 0.0)
    out[:, 3] = np.where(best > 0, (best - second[grp]) / np.maximum(best, 1e-6), 0.0)
    out[:, 4] = size
    out[:, 5] = np.where(score > 0, (score - nxt) / np.maximum(score, 1e-6), 0.0)
    return out


def pair_features(TQ: Table, TS: Table, info: dict, qa: np.ndarray, sa: np.ndarray, rank: np.ndarray,
                  score: np.ndarray, workers: int = -1) -> np.ndarray:
    return np.hstack([
        kernel_features(TQ, TS, qa, sa, info),
        string_features(TQ, TS, qa, sa, workers),
        flag_features(TQ, TS, qa, sa),
        retrieval_features(qa, rank, score),
    ]).astype(np.float32)


SIB_COLS = ["sib_n", "sib_n_s2", "sib_n_s3", "sib_max_p", "sib_mean_p", "sib_n_wjacc", "sib_a_wjacc",
            "sib_num_jacc", "sib_num_near", "sib_unit", "sib_n_tset", "sib_a_tset", "sib_best"]


def sibling_features(TQ: Table, info: dict, pair_q: np.ndarray, pair_s: np.ndarray, best_s: np.ndarray,
                     best_p: np.ndarray, tau: float, cap: int = 10, workers: int = -1) -> np.ndarray:
    """For a pair (q, s): the other records whose confident best Source-1 match
    is s (its cluster), and how similar q is to them."""
    from rapidfuzz import fuzz

    n = len(pair_q)
    out = np.zeros((n, len(SIB_COLS)), np.float32)
    conf = np.flatnonzero((best_s >= 0) & (best_p >= tau))
    if n == 0 or len(conf) == 0:
        return out
    order = conf[np.lexsort((-best_p[conf], best_s[conf]))]
    grp_s = best_s[order]
    uniq_s, first = np.unique(grp_s, return_index=True)
    counts = np.diff(np.append(first, len(order)))
    loc = np.searchsorted(uniq_s, pair_s)
    loc[loc >= len(uniq_s)] = 0
    has = uniq_s[loc] == pair_s
    start = np.where(has, first[loc], 0)
    size = np.where(has, np.minimum(counts[loc], cap + 1), 0)
    rep = np.repeat(np.arange(n), size)
    offs = np.arange(len(rep)) - np.repeat(np.cumsum(size) - size, size)
    mem = order[np.repeat(start, size) + offs]
    self_m = mem == pair_q[rep]
    rep, mem = rep[~self_m], mem[~self_m]
    if len(rep) == 0:
        return out
    k = kernel_features(TQ, TQ, pair_q[rep], mem, info)
    col = {c: i for i, c in enumerate(KERNEL_COLS)}
    mp = best_p[mem]
    n_tset = _cp(fuzz.token_set_ratio, TQ.core[pair_q[rep]], TQ.core[mem], workers)
    a_tset = _cp(fuzz.token_set_ratio, TQ.ad[pair_q[rep]], TQ.ad[mem], workers)
    combo = 0.5 * k[:, col["n_wjacc"]] + 0.3 * k[:, col["a_wjacc"]] + 0.2 * k[:, col["num_jacc"]]
    np.add.at(out[:, 0], rep, 1.0)
    np.add.at(out[:, 1], rep, 1.0 - TQ.s3[mem])
    np.add.at(out[:, 2], rep, TQ.s3[mem].astype(np.float32))
    np.maximum.at(out[:, 3], rep, mp)
    np.add.at(out[:, 4], rep, mp)
    np.maximum.at(out[:, 5], rep, k[:, col["n_wjacc"]])
    np.maximum.at(out[:, 6], rep, k[:, col["a_wjacc"]])
    np.maximum.at(out[:, 7], rep, k[:, col["num_jacc"]])
    np.maximum.at(out[:, 8], rep, k[:, col["num_near"]])
    np.maximum.at(out[:, 9], rep, k[:, col["unit_shared"]])
    np.maximum.at(out[:, 10], rep, n_tset)
    np.maximum.at(out[:, 11], rep, a_tset)
    np.maximum.at(out[:, 12], rep, combo)
    nz = out[:, 0] > 0
    out[nz, 4] /= out[nz, 0]
    return out
