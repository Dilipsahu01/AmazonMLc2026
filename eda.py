"""
Amazon ML Challenge 2026 — Exploratory Data Analysis
=====================================================
Quick EDA to understand dataset characteristics before building the pipeline.

Key questions:
1. Dataset sizes (S1, S2, S3 row counts)
2. Country distribution
3. Singleton ratio in ground truth
4. Match density (avg matches per S1 entity)
5. Non-Latin script prevalence
6. Missing/empty field rates
7. Sample matched pairs (to see actual noise)
"""

import pandas as pd
import numpy as np
import re
import unicodedata
import sys
import time

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = "6ab10eb3b23ba_student_resource/student_resource"
TRAIN_DIR = f"{BASE}/dataset/train"
TEST_DIR = f"{BASE}/dataset/test"

# ─── 1. Load data & basic counts ────────────────────────────────────────────
print("=" * 70)
print("1. LOADING DATA & BASIC COUNTS")
print("=" * 70)

t0 = time.time()
s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t")
print(f"  Train S1: {len(s1):>10,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t")
print(f"  Train S2: {len(s2):>10,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t")
print(f"  Train S3: {len(s3):>10,} rows  ({time.time()-t0:.1f}s)")

t0 = time.time()
gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t")
print(f"  Ground Truth: {len(gt):>10,} rows  ({time.time()-t0:.1f}s)")

# Load test for size comparison
t0 = time.time()
ts1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t")
ts2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t")
ts3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t")
print(f"  Test S1: {len(ts1):>10,} rows")
print(f"  Test S2: {len(ts2):>10,} rows")
print(f"  Test S3: {len(ts3):>10,} rows  ({time.time()-t0:.1f}s)")

print(f"\n  Total train S2+S3: {len(s2)+len(s3):,}")
print(f"  Total test  S2+S3: {len(ts2)+len(ts3):,}")

# ─── 2. Country Distribution ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. COUNTRY DISTRIBUTION")
print("=" * 70)

for name, df in [("Train S1", s1), ("Train S2", s2), ("Train S3", s3),
                  ("Test S1", ts1), ("Test S2", ts2), ("Test S3", ts3)]:
    counts = df["country"].value_counts()
    total = len(df)
    print(f"\n  {name}:")
    for country, cnt in counts.items():
        print(f"    {country:>10s}: {cnt:>10,}  ({cnt/total*100:5.1f}%)")

# ─── 3. Singleton Ratio & Match Density ─────────────────────────────────────
print("\n" + "=" * 70)
print("3. SINGLETON RATIO & MATCH DENSITY")
print("=" * 70)

# Parse matched_entity_ids
gt["match_list"] = gt["matched_entity_ids"].apply(
    lambda x: x.split(",") if pd.notna(x) and str(x).strip() != "" else []
)
gt["num_matches"] = gt["match_list"].apply(len)

total_s1 = len(gt)
singletons = (gt["num_matches"] == 0).sum()
non_singletons = total_s1 - singletons

print(f"  Total S1 entities:    {total_s1:>10,}")
print(f"  Singletons (0 match): {singletons:>10,}  ({singletons/total_s1*100:5.1f}%)")
print(f"  Non-singletons:       {non_singletons:>10,}  ({non_singletons/total_s1*100:5.1f}%)")

print(f"\n  Match count distribution (non-singleton):")
match_dist = gt[gt["num_matches"] > 0]["num_matches"].describe()
print(f"    Mean:   {match_dist['mean']:.2f}")
print(f"    Median: {match_dist['50%']:.1f}")
print(f"    Min:    {match_dist['min']:.0f}")
print(f"    Max:    {match_dist['max']:.0f}")
print(f"    Std:    {match_dist['std']:.2f}")

print(f"\n  Full match count histogram:")
hist = gt["num_matches"].value_counts().sort_index()
for n_matches, count in hist.head(15).items():
    bar = "█" * min(int(count / total_s1 * 200), 60)
    print(f"    {n_matches:>3d} matches: {count:>10,}  ({count/total_s1*100:5.1f}%)  {bar}")
if len(hist) > 15:
    remaining = hist.iloc[15:].sum()
    print(f"    15+ matches: {remaining:>10,}  ({remaining/total_s1*100:5.1f}%)")

# How many have S2-only, S3-only, or both?
def match_sources(match_list):
    has_s2 = any(m.startswith("S2-") for m in match_list)
    has_s3 = any(m.startswith("S3-") for m in match_list)
    if has_s2 and has_s3:
        return "both"
    elif has_s2:
        return "S2_only"
    elif has_s3:
        return "S3_only"
    else:
        return "none"

gt["match_source"] = gt["match_list"].apply(match_sources)
src_dist = gt["match_source"].value_counts()
print(f"\n  Match source breakdown:")
for src, cnt in src_dist.items():
    print(f"    {src:>10s}: {cnt:>10,}  ({cnt/total_s1*100:5.1f}%)")

# ─── 4. Non-Latin Script Detection ──────────────────────────────────────────
print("\n" + "=" * 70)
print("4. NON-LATIN SCRIPT DETECTION")
print("=" * 70)

def has_non_latin(text):
    """Check if text contains non-Latin characters (excluding common punctuation)."""
    if pd.isna(text):
        return False
    for char in str(text):
        if char.isalpha():
            cat = unicodedata.category(char)
            name = unicodedata.name(char, "")
            if not (name.startswith("LATIN") or cat == "Mn"):
                return True
    return False

def detect_scripts(text):
    """Detect which scripts are present in text."""
    if pd.isna(text):
        return set()
    scripts = set()
    for char in str(text):
        if char.isalpha():
            name = unicodedata.name(char, "UNKNOWN")
            script = name.split()[0]
            scripts.add(script)
    return scripts

# Sample for speed (checking all 10M rows would be slow)
sample_size = 50000
print(f"  (Sampling {sample_size:,} rows per source for speed)\n")

for name, df in [("Train S1", s1), ("Train S2", s2), ("Train S3", s3)]:
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    name_non_latin = sample["business_name"].apply(has_non_latin).sum()
    addr_non_latin = sample["business_address"].apply(has_non_latin).sum()
    
    print(f"  {name} (sample of {len(sample):,}):")
    print(f"    Non-Latin in business_name:    {name_non_latin:>6,}  ({name_non_latin/len(sample)*100:5.1f}%)")
    print(f"    Non-Latin in business_address: {addr_non_latin:>6,}  ({addr_non_latin/len(sample)*100:5.1f}%)")
    
    # Detect which scripts
    all_scripts = set()
    for text in sample["business_name"].dropna().head(5000):
        all_scripts |= detect_scripts(text)
    all_scripts.discard("LATIN")
    if all_scripts:
        print(f"    Scripts found (in names): {sorted(all_scripts)}")
    print()

# Same for test
for name, df in [("Test S1", ts1), ("Test S2", ts2), ("Test S3", ts3)]:
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    name_non_latin = sample["business_name"].apply(has_non_latin).sum()
    addr_non_latin = sample["business_address"].apply(has_non_latin).sum()
    print(f"  {name} (sample of {len(sample):,}):")
    print(f"    Non-Latin in business_name:    {name_non_latin:>6,}  ({name_non_latin/len(sample)*100:5.1f}%)")
    print(f"    Non-Latin in business_address: {addr_non_latin:>6,}  ({addr_non_latin/len(sample)*100:5.1f}%)")
    print()

# ─── 5. Missing / Empty Fields ──────────────────────────────────────────────
print("=" * 70)
print("5. MISSING / EMPTY FIELDS")
print("=" * 70)

for name, df in [("Train S1", s1), ("Train S2", s2), ("Train S3", s3),
                  ("Test S1", ts1), ("Test S2", ts2), ("Test S3", ts3)]:
    print(f"\n  {name}:")
    for col in ["business_name", "business_address", "country"]:
        null_count = df[col].isna().sum()
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        total = len(df)
        print(f"    {col:>20s}:  null={null_count:>8,} ({null_count/total*100:.2f}%)  empty={empty_count:>8,} ({empty_count/total*100:.2f}%)")

# ─── 6. Sample Matched Pairs (see actual noise) ─────────────────────────────
print("\n" + "=" * 70)
print("6. SAMPLE MATCHED PAIRS (First 5 non-singleton S1 entities)")
print("=" * 70)

s2_lookup = s2.set_index("entity_id")
s3_lookup = s3.set_index("entity_id")

non_singleton_gt = gt[gt["num_matches"] > 0].head(5)

for _, row in non_singleton_gt.iterrows():
    s1_id = row["source1_entity_id"]
    s1_record = s1[s1["entity_id"] == s1_id].iloc[0]
    
    print(f"\n  S1: {s1_id}")
    print(f"    Name:    {s1_record['business_name']}")
    print(f"    Address: {s1_record['business_address']}")
    print(f"    Country: {s1_record['country']}")
    
    for match_id in row["match_list"]:
        if match_id.startswith("S2-"):
            if match_id in s2_lookup.index:
                rec = s2_lookup.loc[match_id]
                print(f"  ↳ {match_id}")
                print(f"      Name:    {rec['business_name']}")
                print(f"      Address: {rec['business_address']}")
                print(f"      Country: {rec['country']}")
        elif match_id.startswith("S3-"):
            if match_id in s3_lookup.index:
                rec = s3_lookup.loc[match_id]
                print(f"  ↳ {match_id}")
                print(f"      Name:    {rec['business_name']}")
                print(f"      Address: {rec['business_address']}")
                print(f"      Country: {rec['country']}")

# ─── 7. Country-specific match analysis ─────────────────────────────────────
print("\n" + "=" * 70)
print("7. COUNTRY-SPECIFIC MATCH STATS")
print("=" * 70)

s1_country = s1.set_index("entity_id")["country"]
gt["s1_country"] = gt["source1_entity_id"].map(s1_country)

for country in gt["s1_country"].unique():
    country_gt = gt[gt["s1_country"] == country]
    total = len(country_gt)
    singletons_c = (country_gt["num_matches"] == 0).sum()
    avg_matches = country_gt[country_gt["num_matches"] > 0]["num_matches"].mean()
    print(f"\n  {country}:")
    print(f"    Total S1 entities:    {total:>10,}")
    print(f"    Singletons:           {singletons_c:>10,}  ({singletons_c/total*100:5.1f}%)")
    print(f"    Avg matches (non-0):  {avg_matches:>10.2f}")

# ─── 8. Address length stats ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. ADDRESS LENGTH STATS")
print("=" * 70)

for name, df in [("Train S1", s1), ("Train S2", s2), ("Train S3", s3)]:
    addr_len = df["business_address"].astype(str).str.len()
    name_len = df["business_name"].astype(str).str.len()
    print(f"\n  {name}:")
    print(f"    Name length:    mean={name_len.mean():.0f}  median={name_len.median():.0f}  max={name_len.max():.0f}")
    print(f"    Address length: mean={addr_len.mean():.0f}  median={addr_len.median():.0f}  max={addr_len.max():.0f}")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
