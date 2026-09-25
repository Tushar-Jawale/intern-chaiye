"""
Phase 1 - Exploratory Data Analysis
Understand data shape, nulls, noise patterns, country distribution, 
ground truth statistics, and sample records.
"""
import pandas as pd
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = r"f:\TECH\ML challenge\student_resource\dataset"

# ── 1. Load data ──────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_s1 = pd.read_csv(os.path.join(BASE, "train", "train_source1.tsv"), sep="\t")
train_s2 = pd.read_csv(os.path.join(BASE, "train", "train_source2.tsv"), sep="\t")
train_s3 = pd.read_csv(os.path.join(BASE, "train", "train_source3.tsv"), sep="\t")
train_gt = pd.read_csv(os.path.join(BASE, "train", "train_ground_truth.tsv"), sep="\t")

test_s1 = pd.read_csv(os.path.join(BASE, "test", "test_source1.tsv"), sep="\t")
test_s2 = pd.read_csv(os.path.join(BASE, "test", "test_source2.tsv"), sep="\t")
test_s3 = pd.read_csv(os.path.join(BASE, "test", "test_source3.tsv"), sep="\t")

print("All files loaded successfully!\n")

# ── 2. Shape & columns ──────────────────────────────────────
print("=" * 70)
print("DATASET SHAPES & COLUMNS")
print("=" * 70)
for name, df in [("train_s1", train_s1), ("train_s2", train_s2), ("train_s3", train_s3),
                  ("train_gt", train_gt),
                  ("test_s1", test_s1), ("test_s2", test_s2), ("test_s3", test_s3)]:
    print(f"{name:15s} | {df.shape[0]:>10,} rows x {df.shape[1]} cols | Columns: {list(df.columns)}")
print()

# ── 3. Null analysis ────────────────────────────────────────
print("=" * 70)
print("NULL / MISSING VALUE ANALYSIS")
print("=" * 70)
for name, df in [("train_s1", train_s1), ("train_s2", train_s2), ("train_s3", train_s3),
                  ("test_s1", test_s1), ("test_s2", test_s2), ("test_s3", test_s3)]:
    nulls = df.isnull().sum()
    pcts = (df.isnull().sum() / len(df) * 100).round(2)
    print(f"\n{name}:")
    for col in df.columns:
        print(f"  {col:25s} | {nulls[col]:>8,} nulls ({pcts[col]:>6.2f}%)")
print()

# ── 4. Country distribution ─────────────────────────────────
print("=" * 70)
print("COUNTRY DISTRIBUTION")
print("=" * 70)
for name, df in [("train_s1", train_s1), ("train_s2", train_s2), ("train_s3", train_s3),
                  ("test_s1", test_s1), ("test_s2", test_s2), ("test_s3", test_s3)]:
    print(f"\n{name}:")
    vc = df['country'].value_counts(dropna=False)
    for country, count in vc.items():
        print(f"  {str(country):20s} | {count:>10,} ({count/len(df)*100:.1f}%)")
print()

# ── 5. Ground truth analysis ────────────────────────────────
print("=" * 70)
print("GROUND TRUTH ANALYSIS")
print("=" * 70)
print(f"Total S1 entities in ground truth: {len(train_gt):,}")

singletons = train_gt['matched_entity_ids'].isna().sum()
print(f"Singletons (no matches):           {singletons:,} ({singletons/len(train_gt)*100:.1f}%)")
print(f"With matches:                      {len(train_gt) - singletons:,} ({(len(train_gt)-singletons)/len(train_gt)*100:.1f}%)")

non_singleton = train_gt[train_gt['matched_entity_ids'].notna()].copy()
non_singleton['match_count'] = non_singleton['matched_entity_ids'].str.split(',').str.len()
print(f"\nMatch count distribution (non-singletons):")
print(f"  Mean:   {non_singleton['match_count'].mean():.2f}")
print(f"  Median: {non_singleton['match_count'].median():.1f}")
print(f"  Max:    {non_singleton['match_count'].max()}")
print(f"  Distribution:")
vc = non_singleton['match_count'].value_counts().sort_index()
for count, freq in vc.head(15).items():
    print(f"    {count} matches | {freq:,} entities")
if len(vc) > 15:
    print(f"    ... ({len(vc)} distinct counts total)")

all_matches = non_singleton['matched_entity_ids'].str.split(',').explode()
s2_matches = all_matches[all_matches.str.startswith('S2-')].shape[0]
s3_matches = all_matches[all_matches.str.startswith('S3-')].shape[0]
print(f"\nMatch source breakdown:")
print(f"  S2 matches: {s2_matches:,}")
print(f"  S3 matches: {s3_matches:,}")
print(f"  Total:      {s2_matches + s3_matches:,}")
print()

# ── 6. Sample records ───────────────────────────────────────
print("=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

for src_name, df in [("train_source1", train_s1), ("train_source2", train_s2), ("train_source3", train_s3)]:
    print(f"\n--- {src_name} (first 3) ---")
    for _, row in df.head(3).iterrows():
        print(f"  ID: {row['entity_id']}  |  Name: {row['business_name']}  |  Addr: {str(row['business_address'])[:80]}  |  Country: {row['country']}")

# ── 7. Matched pair examples ─────────────────────────────────
print("\n" + "=" * 70)
print("MATCHED PAIR EXAMPLES (understanding noise patterns)")
print("=" * 70)

s2_lookup = train_s2.set_index('entity_id').to_dict('index')
s3_lookup = train_s3.set_index('entity_id').to_dict('index')
s1_lookup = train_s1.set_index('entity_id').to_dict('index')

shown = 0
for _, row in non_singleton.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_lookup:
        continue
    s1_rec = s1_lookup[s1_id]
    matched_ids = row['matched_entity_ids'].split(',')
    for mid in matched_ids[:1]:
        mid = mid.strip()
        if mid.startswith('S2-') and mid in s2_lookup:
            m_rec = s2_lookup[mid]
        elif mid.startswith('S3-') and mid in s3_lookup:
            m_rec = s3_lookup[mid]
        else:
            continue
        print(f"\n  S1 [{s1_id}]:")
        print(f"    Name:    {s1_rec['business_name']}")
        print(f"    Address: {str(s1_rec['business_address'])[:120]}")
        print(f"    Country: {s1_rec['country']}")
        print(f"  MATCHED -> [{mid}]:")
        print(f"    Name:    {m_rec['business_name']}")
        print(f"    Address: {str(m_rec['business_address'])[:120]}")
        print(f"    Country: {m_rec['country']}")
        print(f"  {'_' * 60}")
        shown += 1
    if shown >= 15:
        break

# ── 8. Text length stats ─────────────────────────────────────
print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS")
print("=" * 70)
for name, df in [("train_s1", train_s1), ("train_s2", train_s2), ("train_s3", train_s3)]:
    name_lens = df['business_name'].fillna('').str.len()
    addr_lens = df['business_address'].fillna('').str.len()
    print(f"\n{name}:")
    print(f"  name length    | mean={name_lens.mean():.0f}, median={name_lens.median():.0f}, min={name_lens.min()}, max={name_lens.max()}")
    print(f"  address length | mean={addr_lens.mean():.0f}, median={addr_lens.median():.0f}, min={addr_lens.min()}, max={addr_lens.max()}")

# ── 9. France samples (test only) ────────────────────────────
print("\n" + "=" * 70)
print("FRANCE SAMPLES (test set)")
print("=" * 70)
for name, df in [("test_s1", test_s1), ("test_s2", test_s2), ("test_s3", test_s3)]:
    france = df[df['country'].str.lower().str.contains('france', na=False)]
    if len(france) > 0:
        print(f"\n{name} - France records: {len(france):,}")
        for _, row in france.head(3).iterrows():
            print(f"  ID: {row['entity_id']}  |  Name: {row['business_name']}  |  Addr: {str(row['business_address'])[:100]}  |  Country: {row['country']}")
    else:
        print(f"\n{name} - No France records found")

print("\n\nDONE - EDA Complete!")
