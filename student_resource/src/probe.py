"""
Per-country leaderboard probes from an existing matching_results.tsv.

Blanking a country's rows turns every one of its entities into "no match":
such an entity scores 1 if it truly has no match and 0 otherwise. With w the
country's share of Source-1 entities and s the share of its entities with no
true match (5.5% in train), the leaderboard change isolates that country:

    F_country = s + (LB_full - LB_blank) / w

    python src/probe.py make --matching output/matching_results.tsv --test-dir dataset/test --blank France
    python src/probe.py solve --test-dir dataset/test --country France --lb-full 0.967 --lb-blank 0.83
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter

EMPTY_SHARE = 0.055


def countries(test_dir: str) -> dict[str, str]:
    out = {}
    with open(os.path.join(test_dir, "test_source1.tsv"), encoding="utf-8") as h:
        head = h.readline().rstrip("\n").split("\t")
        ci = head.index("country")
        for line in h:
            f = line.rstrip("\n").split("\t")
            if len(f) > ci and f[0]:
                out[f[0]] = f[ci].strip()
    return out


def make(a: argparse.Namespace) -> None:
    cmap = countries(a.test_dir)
    blank = {c for c in a.blank.split(",") if c}
    out = a.out or os.path.join(os.path.dirname(a.matching) or ".", "probe_blank_" + "_".join(sorted(blank)))
    os.makedirs(out, exist_ok=True)
    kept = Counter()
    emptied = Counter()
    with open(a.matching, encoding="utf-8") as src, \
            open(os.path.join(out, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as dst:
        dst.write(src.readline())
        for line in src:
            eid, _, ids = line.rstrip("\n").partition("\t")
            c = cmap.get(eid, "")
            if c in blank:
                emptied[c] += 1 if ids else 0
                dst.write(eid + "\t\n")
            else:
                kept[c] += 1 if ids else 0
                dst.write(line if line.endswith("\n") else line + "\n")
    if a.candidate:
        shutil.copy(a.candidate, os.path.join(out, "candidate_pairs.tsv"))
    total = Counter(cmap.values())
    n = sum(total.values())
    for c in sorted(total):
        tag = "BLANKED" if c in blank else "kept"
        rows = emptied[c] if c in blank else kept[c]
        print(f"  {c:8s} {tag:8s} entities {total[c]:,} ({total[c] / n:.4f} of all), rows with matches {rows:,}")
    print(f"wrote {out}/matching_results.tsv")


def solve(a: argparse.Namespace) -> None:
    total = Counter(countries(a.test_dir).values())
    w = total[a.country] / sum(total.values())
    f = a.empty_share + (a.lb_full - a.lb_blank) / w
    rest = (a.lb_full - w * f) / (1 - w)
    print(f"{a.country}: share of entities {w:.4f}  ->  F0.5 {f:.4f}   (all other countries together {rest:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--matching", required=True)
    m.add_argument("--candidate", default="")
    m.add_argument("--test-dir", required=True)
    m.add_argument("--blank", required=True, help="comma list of countries to empty")
    m.add_argument("--out", default="")
    s = sub.add_parser("solve")
    s.add_argument("--test-dir", required=True)
    s.add_argument("--country", required=True)
    s.add_argument("--lb-full", type=float, required=True)
    s.add_argument("--lb-blank", type=float, required=True)
    s.add_argument("--empty-share", type=float, default=EMPTY_SHARE)
    a = p.parse_args()
    make(a) if a.cmd == "make" else solve(a)


if __name__ == "__main__":
    main()
