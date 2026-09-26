"""
Normalization v3.

Names and addresses are reduced to a few string fields that the retrieval keys
and pair features read. Differences from preprocess.py:

  * Indic-script tokens are mapped to the Latin word they stand for with a
    dictionary learned from training pairs (a Source 2/3 name in Devanagari,
    Telugu, ... next to its Source 1 owner written in Latin). Tokens never seen
    in training fall back to a phonetic romanization. All Indic blocks share
    the Devanagari layout, so one table covers every script.
  * Numbers lose leading zeros and ordinal suffixes (0684 = 684, 184th = 184).
  * Unit ids keep their letter (L 378 -> l378, AF-0684 -> af684, 41 C -> 41c).
  * A comma component that is a state name or code becomes one state code.
  * Web-domain names (dphealth.com) are flagged and reduced to their label.
  * Street-type abbreviations are expanded per country label; unknown labels
    get only the unambiguous shared table.

    python src/norm3.py learn --data dataset --work work3
    python src/norm3.py build --data dataset --work work3
    python src/norm3.py self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FILES = [
    ("train", "train_source1.tsv", "train_s1"),
    ("train", "train_source2.tsv", "train_s2"),
    ("train", "train_source3.tsv", "train_s3"),
    ("test", "test_source1.tsv", "test_s1"),
    ("test", "test_source2.tsv", "test_s2"),
    ("test", "test_source3.tsv", "test_s3"),
]
OUT_COLS = [
    "entity_id", "country", "nm", "core", "phon", "sfx", "dom", "scr",
    "ad", "st", "nums", "units", "pin",
]

_INDIC_RE = re.compile("[\u0900-\u0DFF]")
_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)
_SPLIT = re.compile(r"[\s,;:/\\()\[\]{}<>\"`~!@#$%^*+=?|_\-\u2013\u2014\u00b7\u2022\u2026\u201c\u201d\u2018\u2019'.\u00b0\u00ba\u2116]+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:co\.in|com|net|org|in|co|biz|info|fr|us|io)/?$")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|eme|e)\b")
_HYPH_L = re.compile(r"\b([A-Za-z]{1,2})-0*(\d+)\b")
_HYPH_SKIP = {"no", "nr", "ph", "st"}
_HYPH_R = re.compile(r"\b0*(\d+)-([A-Za-z])\b")
_NULLS = {"nan", "null", "<null>", "none", "n/a", "na"}
# Single letters that are words in that country's addresses, never unit letters.
_NOT_UNIT = {"us": "nsew", "france": "nsewrdl"}
_UNIT_RX: dict[str, tuple[re.Pattern, re.Pattern]] = {}


def _unit_rx(country: str) -> tuple[re.Pattern, re.Pattern]:
    rx = _UNIT_RX.get(country)
    if rx is None:
        banned = set(_NOT_UNIT.get(country, "nsew"))
        letters = "".join(ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch not in banned)
        rx = (re.compile(rf"\b([{letters}])\s*0*(\d+)\b"), re.compile(rf"\b0*(\d+)\s?([{letters}])\b"))
        _UNIT_RX[country] = rx
    return rx

# ---------------------------------------------------------------- Indic -----

_V = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
      0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
      0x60: "ri", 0x61: "li"}
_C = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
      0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
      0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
      0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
      0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
      0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_CF = {0x4E: "t", 0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}
_M = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
      0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x56: "ai",
      0x57: "au", 0x62: "li", 0x63: "li"}
_NUKTA = {"ph": "f", "j": "z", "k": "q", "d": "r", "dh": "rh"}


def has_indic(text: str) -> bool:
    return bool(text) and _INDIC_RE.search(text) is not None


def dnorm(token: str) -> str:
    """Same token written in the Devanagari block (blocks share one layout)."""
    return "".join(chr(0x0900 + (ord(ch) & 0x7F)) if 0x0900 <= ord(ch) <= 0x0DFF else ch for ch in token)


def romanize(token: str) -> str:
    out: list[str] = []
    pend = False
    for ch in token:
        o = ord(ch)
        if not 0x0900 <= o <= 0x0DFF:
            if pend:
                out.append("a")
                pend = False
            out.append(ch)
            continue
        off = o & 0x7F
        if off in _C:
            if pend:
                out.append("a")
            out.append(_C[off])
            pend = True
        elif off in _M:
            out.append(_M[off])
            pend = False
        elif off == 0x4D:
            pend = False
        elif off == 0x3C:
            if pend and out:
                out[-1] = _NUKTA.get(out[-1], out[-1])
        elif off in _V:
            if pend:
                out.append("a")
                pend = False
            out.append(_V[off])
        elif off in (0x01, 0x02, 0x70):
            if pend:
                out.append("a")
                pend = False
            out.append("n")
        elif off == 0x03:
            if pend:
                out.append("a")
                pend = False
            out.append("h")
        elif 0x66 <= off <= 0x6F:
            if pend:
                out.append("a")
                pend = False
            out.append(str(off - 0x66))
        elif off in _CF:
            if pend:
                out.append("a")
            out.append(_CF[off])
            pend = False
    return "".join(out).lower()


def phon(word: str) -> str:
    """Consonant skeleton. English spelling and romanized Indic meet here."""
    w = word.lower()
    if not w:
        return ""
    w = w.replace("chh", "ch").replace("ch", "#").replace("ck", "k")
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"g(?=[ei])", "j", w)
    for a, b in (("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
                 ("gh", "g"), ("jh", "j"), ("c", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s")):
        w = w.replace(a, b)
    if w.endswith("j"):
        w = w[:-1] + "s"
    first, rest = w[0], w[1:]
    if first in "aeiouy":
        first = ""
    rest = re.sub(r"[aeiouyh]", "", rest)
    out: list[str] = [first] if first else []
    for ch in rest:
        if not out or ch != out[-1]:
            out.append(ch)
    return "".join(out).replace("#", "c")


# ----------------------------------------------------------- tables ---------

LEGAL = {
    "pvt": "private", "private": "private", "priv": "private", "pvtltd": "private",
    "ltd": "limited", "limited": "limited", "ltda": "limited", "limted": "limited",
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "plc": "plc", "pc": "pc", "pa": "pa",
    "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "scop": "scop", "ei": "ei", "eirl": "eirl", "scp": "scp", "scs": "scs",
    "selarl": "selarl", "gie": "gie", "ets": "ets", "etablissements": "ets",
}
NAME_STOP = {"the", "of", "and", "for", "in", "at", "by", "a", "an", "to", "on", "dba", "aka",
             "du", "de", "des", "la", "le", "les", "et", "d", "l", "da", "del", "y"}
ADDR_COMMON = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "ln": "lane", "pkwy": "parkway", "hwy": "highway", "trl": "trail", "sq": "square",
    "apt": "apartment", "apts": "apartment", "appt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "opp": "opposite", "nr": "near", "mkt": "market", "sec": "sector",
    "sect": "sector", "indl": "industrial", "soc": "society", "dist": "district",
    "distt": "district", "vill": "village", "no": "", "number": "", "pin": "", "pincode": "",
    "null": "", "nan": "", "none": "",
}
ADDR_BY_COUNTRY = {
    "us": {"st": "street", "str": "street", "ste": "suite", "dr": "drive", "ct": "court", "pl": "place",
           "cir": "circle", "ter": "terrace", "terr": "terrace", "n": "north", "s": "south",
           "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
           "sw": "southwest", "mt": "mount", "ft": "fort", "pt": "point", "hts": "heights",
           "jct": "junction", "expy": "expressway", "fwy": "freeway", "cv": "cove", "xing": "crossing",
           "rte": "route", "ctr": "center", "cntr": "center", "unit": "", "suite": "suite"},
    "india": {"st": "street", "dr": "doctor", "hno": "house", "h": "house", "ind": "industrial",
              "est": "estate", "chs": "society", "bangalore": "bengaluru", "bombay": "mumbai",
              "madras": "chennai", "calcutta": "kolkata", "gurgaon": "gurugram", "poona": "pune",
              "mysore": "mysuru", "trivandrum": "thiruvananthapuram", "baroda": "vadodara",
              "cochin": "kochi", "mangalore": "mangaluru", "belgaum": "belagavi",
              "pondicherry": "puducherry", "allahabad": "prayagraj", "benares": "varanasi",
              "simla": "shimla", "hubli": "hubballi", "orissa": "odisha", "tal": "taluka",
              "po": "post", "ps": "police", "vpo": "village"},
    "france": {"r": "rue", "st": "saint", "ste": "sainte", "bld": "boulevard", "boul": "boulevard",
               "pl": "place", "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route",
               "all": "allee", "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours",
               "res": "residence", "bat": "batiment", "n": "", "d": "", "l": "", "de": "", "du": "",
               "des": "", "la": "", "le": "", "les": "", "cedex": ""},
}
_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
_IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
_IN_CODES = {"ap", "ar", "as", "br", "cg", "ct", "ga", "gj", "hr", "hp", "jh", "ka", "kl", "mp", "mh",
             "mn", "ml", "mz", "nl", "od", "or", "pb", "rj", "sk", "tn", "ts", "tg", "tr", "up", "uk",
             "ut", "wb", "dl", "jk", "ch", "py"}
_IN_ALIAS = {"or": "od", "ct": "cg", "tg": "ts", "ut": "uk"}
_FR_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq", "pays de la loire": "pdl",
    "ile de france": "idf", "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "occitanie": "occ", "provence alpes cote d azur": "pac", "auvergne rhone alpes": "ara",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
    # Source 2/3 often name the departement where Source 1 names the region.
    "gironde": "naq", "loire atlantique": "pdl", "nord": "hdf", "pas de calais": "hdf",
}


def _state_table(country: str) -> dict[str, str]:
    if country == "us":
        table = dict(_US_STATES)
        table.update({code: code for code in _US_STATES.values()})
        return table
    if country == "india":
        table = dict(_IN_STATES)
        table.update({code: _IN_ALIAS.get(code, code) for code in _IN_CODES})
        return table
    if country == "france":
        return dict(_FR_REGIONS)
    return {}


_STATES = {c: _state_table(c) for c in ("us", "india", "france")}
_STATE_PHON = {c: {phon(k.replace(" ", "")): v for k, v in t.items() if len(k) > 3} for c, t in _STATES.items()}

# ------------------------------------------------------- normalization ------

_TR: dict[str, dict[str, str]] = {"name": {}, "name_d": {}, "addr": {}, "addr_d": {}, "vphon": {}}


def set_translit(table: dict | None) -> None:
    global _TR
    base = {"name": {}, "name_d": {}, "addr": {}, "addr_d": {}, "vphon": {}}
    if table:
        base.update(table)
    _TR = base


def _fold(token: str) -> str:
    if token.isascii():
        return token.lower()
    return unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii").lower()


def translit(token: str, kind: str) -> str:
    token = token.translate(_ZW)
    got = _TR[kind].get(token)
    if got is None:
        got = _TR[kind + "_d"].get(dnorm(token))
    if got is None:
        got = _fold(romanize(token))
        if kind == "name":
            key = phon(got)
            if len(key) >= 3:
                got = _TR["vphon"].get(key, got)
    return got


def _raw_tokens(text: str, kind: str) -> tuple[list[str], bool]:
    """Split on punctuation, map Indic tokens, fold accents. Returns (tokens, had_indic)."""
    indic = has_indic(text)
    out: list[str] = []
    for tok in _SPLIT.split(text):
        if not tok:
            continue
        if indic and has_indic(tok):
            tok = translit(tok, kind)
            out.extend(t for t in _NON_ALNUM.split(tok) if t)
        else:
            tok = _fold(tok)
            out.extend(t for t in _NON_ALNUM.split(tok) if t)
    return out, indic


def _merge_letters(tokens: list[str]) -> list[str]:
    out: list[str] = []
    run: list[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


_LEET = str.maketrans("013457", "oleast")


def name_tokens(raw: str) -> tuple[list[str], list[str], bool, bool]:
    """Returns (all tokens canonical, legal forms, is_domain, had_indic)."""
    if not isinstance(raw, str):
        return [], [], False, False
    text = raw.strip()
    if not text or text.lower() in _NULLS:
        return [], [], False, False
    low = text.lower()
    m = _DOMAIN.match(low)
    if m:
        label = m.group(1).replace("-", "")
        return [label], [], True, False
    text = text.replace("&", " and ").replace("'", "").replace("\u2019", "")
    toks, indic = _raw_tokens(text, "name")
    toks = _merge_letters(toks)
    out: list[str] = []
    legal: list[str] = []
    for t in toks:
        if not t.isalpha() and not t.isdigit():
            letters = sum(ch.isalpha() for ch in t)
            if letters >= 3 and len(t) - letters <= 2:
                t = t.translate(_LEET)
        canon = LEGAL.get(t)
        if canon is not None:
            legal.append(canon)
            t = canon
        if out and out[-1] == t:
            continue
        out.append(t)
    return out, legal, False, indic


def core_of(tokens: list[str]) -> list[str]:
    core = [t for t in tokens if t not in LEGAL and t not in NAME_STOP and t not in LEGAL.values() and len(t) > 1]
    if not core:
        core = [t for t in tokens if t not in NAME_STOP]
    return core


def norm_name(raw: str) -> dict:
    toks, legal, dom, indic = name_tokens(raw)
    core = core_of(toks)
    return {
        "nm": " ".join(toks),
        "core": " ".join(core),
        "phon": " ".join(p for p in (phon(t) for t in core if not t.isdigit()) if p),
        "sfx": " ".join(sorted(set(legal))),
        "dom": 1 if dom else 0,
        "scr": 1 if indic else 0,
    }


def _hyph_join(m: re.Match) -> str:
    if m.group(1).lower() in _HYPH_SKIP:
        return m.group(1) + " " + m.group(2)
    return m.group(1) + m.group(2)


def _addr_component(comp: str, country: str) -> tuple[list[str], bool]:
    comp = _HYPH_L.sub(_hyph_join, comp.replace("&", " and "))
    comp = _HYPH_R.sub(r"\1\2", comp)
    toks, indic = _raw_tokens(comp, "addr")
    if not toks:
        return [], indic
    left, right = _unit_rx(country)
    s = " ".join(toks)
    s = _ORDINAL.sub(r"\1", s)
    s = left.sub(r"\1\2", s)
    s = right.sub(r"\1\2", s)
    return s.split(), indic


def _match_state(country: str, toks: list[str], indic: bool) -> str:
    table = _STATES.get(country)
    if not table:
        return ""
    key = " ".join(toks)
    code = table.get(key)
    if code:
        return code
    if indic:
        return _STATE_PHON[country].get(phon(key.replace(" ", "")), "")
    return ""


def norm_addr(raw: str, country: str) -> dict:
    empty = {"ad": "", "st": "", "nums": "", "units": "", "pin": ""}
    if not isinstance(raw, str):
        return empty
    text = raw.strip()
    if not text or text.lower() in _NULLS:
        return empty
    text = re.sub(r"<\s*null\s*>", " ", text, flags=re.I)
    abbr = dict(ADDR_COMMON)
    abbr.update(ADDR_BY_COUNTRY.get(country, {}))
    state = ""
    tokens: list[str] = []
    for comp in re.split(r"[,;]+", text):
        toks, indic = _addr_component(comp, country)
        if not toks:
            continue
        code = _match_state(country, toks, indic)
        if code:
            state = code
            continue
        for t in toks:
            t = abbr.get(t, t)
            if t:
                tokens.extend(t.split())
    nums: list[str] = []
    units: list[str] = []
    for t in tokens:
        if t.isdigit():
            n = t.lstrip("0")
            if n and n not in nums:
                nums.append(n)
        elif any(ch.isdigit() for ch in t):
            if t not in units:
                units.append(t)
            for run in re.findall(r"\d+", t):
                n = run.lstrip("0")
                if n and n not in nums:
                    nums.append(n)
    pin = next((n for n in reversed(nums) if len(n) == 6), "")
    return {"ad": " ".join(tokens), "st": state, "nums": " ".join(nums), "units": " ".join(units), "pin": pin}


def norm_chunk(args) -> dict[str, list]:
    ids, countries, names, addrs = args
    out = {c: [] for c in OUT_COLS}
    for eid, ctry, name, addr in zip(ids, countries, names, addrs):
        key = (ctry or "").strip().lower()
        n = norm_name(name)
        a = norm_addr(addr, key)
        out["entity_id"].append(eid)
        out["country"].append(ctry)
        for col in ("nm", "core", "phon", "sfx", "dom", "scr"):
            out[col].append(n[col])
        for col in ("ad", "st", "nums", "units", "pin"):
            out[col].append(a[col])
    return out


def _init_worker(table: dict) -> None:
    set_translit(table)


# --------------------------------------------------------------- learn ------

def _read_tsv(path: str, chunksize: int | None = None, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                       chunksize=chunksize, usecols=usecols)


def _s1_tokens(name: str) -> list[str]:
    toks, _, _, _ = name_tokens(name)
    return toks


def _addr_plain(addr: str) -> list[str]:
    toks, _ = _raw_tokens(addr or "", "addr")
    return toks


def _align(q_raw: str, s1_toks: list[str], positional: bool, counts: dict, min_sim: float) -> None:
    from rapidfuzz import fuzz

    q_toks = [t.translate(_ZW) for t in _SPLIT.split(q_raw.replace("&", " ")) if t]
    if not q_toks or not s1_toks:
        return
    s1_ph = [phon(t) for t in s1_toks]
    same_len = positional and len(q_toks) == len(s1_toks)
    for i, tok in enumerate(q_toks):
        if not has_indic(tok):
            continue
        rom = phon(_fold(romanize(tok)))
        if same_len:
            counts[tok][s1_toks[i]] += 1
            continue
        best, best_j = 0.0, -1
        for j, ph in enumerate(s1_ph):
            sc = fuzz.ratio(rom, ph)
            if sc > best:
                best, best_j = sc, j
        if best_j >= 0 and best >= min_sim:
            counts[tok][s1_toks[best_j]] += 1


def _finish(counts: dict[str, Counter]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    dcounts: dict[str, Counter] = defaultdict(Counter)
    for tok, cnt in counts.items():
        total = sum(cnt.values())
        best, c = cnt.most_common(1)[0]
        if c >= 2 and c / total >= 0.5:
            exact[tok] = best
        dcounts[dnorm(tok)].update(cnt)
    dmap: dict[str, str] = {}
    for tok, cnt in dcounts.items():
        total = sum(cnt.values())
        best, c = cnt.most_common(1)[0]
        if c >= 2 and c / total >= 0.5:
            dmap[tok] = best
    return exact, dmap


def learn_cmd(args: argparse.Namespace) -> None:
    t0 = time.time()
    train = os.path.join(args.data, "train")
    os.makedirs(args.work, exist_ok=True)
    rows: dict[str, tuple[str, str]] = {}
    for fname in ("train_source2.tsv", "train_source3.tsv"):
        for chunk in _read_tsv(os.path.join(train, fname), chunksize=500_000,
                               usecols=["entity_id", "business_name", "business_address"]):
            name_m = chunk["business_name"].str.contains(_INDIC_RE, na=False)
            addr_m = chunk["business_address"].str.contains(_INDIC_RE, na=False)
            sub = chunk[name_m | addr_m]
            for eid, nm, ad in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
                rows[eid] = (nm, ad)
        print(f"  {fname}: {len(rows):,} records with Indic text so far ({time.time() - t0:.0f}s)", flush=True)
    owner: dict[str, str] = {}
    for chunk in _read_tsv(os.path.join(train, "train_ground_truth.tsv"), chunksize=500_000):
        for s1, matched in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if not matched:
                continue
            for mid in matched.split(","):
                if mid in rows:
                    owner[mid] = s1
    need = set(owner.values())
    s1_rec: dict[str, tuple[str, str]] = {}
    vocab: Counter = Counter()
    for chunk in _read_tsv(os.path.join(train, "train_source1.tsv"), chunksize=500_000,
                           usecols=["entity_id", "business_name", "business_address"]):
        sub = chunk[chunk["entity_id"].isin(need)]
        for eid, nm, ad in zip(sub["entity_id"], sub["business_name"], sub["business_address"]):
            s1_rec[eid] = (nm, ad)
        for nm in chunk["business_name"]:
            if nm and nm.isascii():
                vocab.update(t for t in _NON_ALNUM.split(nm.lower()) if len(t) >= 3 and t.isalpha())
    vphon: dict[str, str] = {}
    for tok, c in vocab.most_common():
        if c < 3:
            break
        key = phon(LEGAL.get(tok, tok))
        if len(key) >= 3 and key not in vphon:
            vphon[key] = LEGAL.get(tok, tok)
    print(f"  owned Indic records {len(owner):,}, owners {len(s1_rec):,} ({time.time() - t0:.0f}s)", flush=True)
    name_counts: dict[str, Counter] = defaultdict(Counter)
    addr_counts: dict[str, Counter] = defaultdict(Counter)
    for qid, s1 in owner.items():
        q_name, q_addr = rows[qid]
        s_name, s_addr = s1_rec.get(s1, ("", ""))
        if has_indic(q_name) and s_name:
            _align(q_name, _s1_tokens(s_name), True, name_counts, 50.0)
        if has_indic(q_addr) and s_addr:
            _align(q_addr, _addr_plain(s_addr), False, addr_counts, 60.0)
    name_ex, name_d = _finish(name_counts)
    addr_ex, addr_d = _finish(addr_counts)
    table = {"name": name_ex, "name_d": name_d, "addr": addr_ex, "addr_d": addr_d, "vphon": vphon}
    path = os.path.join(args.work, "translit.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False)
    print(f"Wrote {path}: name {len(name_ex):,}+{len(name_d):,}  addr {len(addr_ex):,}+{len(addr_d):,} "
          f"vocab skeletons {len(vphon):,} in {time.time() - t0:.0f}s", flush=True)
    sample = list(name_ex.items())[:15]
    print("  sample:", sample, flush=True)


def load_translit(work: str) -> dict:
    path = os.path.join(work, "translit.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} missing, Indic text uses the phonetic fallback only", flush=True)
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------- build ------

def build_cmd(args: argparse.Namespace) -> None:
    table = load_translit(args.work)
    set_translit(table)
    out_dir = os.path.join(args.work, "norm")
    os.makedirs(out_dir, exist_ok=True)
    workers = args.workers or max(1, (os.cpu_count() or 2))
    splits = set(args.splits.split(","))
    schema = pa.schema([(c, pa.int8() if c in ("dom", "scr") else pa.string()) for c in OUT_COLS])
    pool = Pool(workers, initializer=_init_worker, initargs=(table,)) if workers > 1 else None
    try:
        for split, fname, stem in FILES:
            if split not in splits:
                continue
            path = os.path.join(out_dir, stem + ".parquet")
            if os.path.exists(path) and not args.force:
                print(f"skip {path}", flush=True)
                continue
            t0 = time.time()
            tmp = path + ".tmp"
            writer = pq.ParquetWriter(tmp, schema)
            n = 0
            for chunk in _read_tsv(os.path.join(args.data, split, fname), chunksize=400_000,
                                   usecols=["entity_id", "business_name", "business_address", "country"]):
                ids = chunk["entity_id"].tolist()
                ctry = chunk["country"].tolist()
                names = chunk["business_name"].tolist()
                addrs = chunk["business_address"].tolist()
                step = max(1, len(ids) // (workers * 4))
                parts = [(ids[i:i + step], ctry[i:i + step], names[i:i + step], addrs[i:i + step])
                         for i in range(0, len(ids), step)]
                results = pool.map(norm_chunk, parts) if pool else [norm_chunk(p) for p in parts]
                cols = {c: [] for c in OUT_COLS}
                for res in results:
                    for c in OUT_COLS:
                        cols[c].extend(res[c])
                writer.write_table(pa.Table.from_pydict(cols, schema=schema))
                n += len(ids)
                print(f"  {stem}: {n:,} rows ({time.time() - t0:.0f}s)", flush=True)
            writer.close()
            os.replace(tmp, path)
            print(f"Wrote {path}", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    gt_out = os.path.join(out_dir, "train_gt.parquet")
    if "train" in splits and (not os.path.exists(gt_out) or args.force):
        gt = _read_tsv(os.path.join(args.data, "train", "train_ground_truth.tsv"))
        gt.to_parquet(gt_out, index=False)
        print(f"Wrote {gt_out}", flush=True)


def self_test() -> None:
    set_translit({})
    assert romanize("परफेक्ट") == "paraphekt", romanize("परफेक्ट")
    assert phon("perfect") == phon(romanize("परफेक्ट")), (phon("perfect"), phon(romanize("परफेक्ट")))
    n = norm_name("Ahuja Pr0jects Pvt. Ltd.")
    assert n["core"] == "ahuja projects" and "private" in n["sfx"] and "limited" in n["sfx"], n
    n = norm_name("dphealth.com")
    assert n["dom"] == 1 and n["core"] == "dphealth", n
    n = norm_name("C.I.T. Colony Traders & Sons")
    assert n["core"].startswith("cit colony traders"), n
    a = norm_addr("AF-0684, NANDGRAM, GHAZIABAD, Uttar Pradesh", "india")
    assert a["st"] == "up" and "af684" in a["units"].split() and "684" in a["nums"].split(), a
    a = norm_addr("14310 184th Place, <NULL>, Renton, WA", "us")
    assert a["st"] == "wa" and a["nums"].split() == ["14310", "184"] and "place" in a["ad"], a
    a = norm_addr("L 378 Shashtri Marg Bararpur, Shahdara, DL", "india")
    assert "l378" in a["units"] and a["st"] == "dl", a
    a = norm_addr("41 C RUE DU MARECHAL FRENCH, ST.-HERBLAIN, Pays de la Loire", "france")
    assert "41c" in a["units"] and "saint" in a["ad"] and a["st"] == "pdl", a
    a = norm_addr("Tamil Nadu, 11, Coimbatore North, 236, Saraswathy Apt", "india")
    assert a["st"] == "tn" and "apartment" in a["ad"], a
    a = norm_addr("B 1601 Ireo Skyon, Gurgaon, हरियाणा", "india")
    assert a["st"] == "hr" and "gurugram" in a["ad"], a
    print("norm3 self-test OK", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalization v3")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-test")
    for name in ("learn", "build"):
        s = sub.add_parser(name)
        s.add_argument("--data", required=True, help="folder with train/ and test/ TSVs")
        s.add_argument("--work", required=True)
        s.add_argument("--workers", type=int, default=0)
        s.add_argument("--splits", default="train,test")
        s.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cli = parse_args()
    if cli.cmd == "self-test":
        self_test()
    elif cli.cmd == "learn":
        learn_cmd(cli)
    else:
        build_cmd(cli)
