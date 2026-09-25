"""
FAST preprocessing runner - uses vectorized operations + multiprocessing.
Processes all train/test datasets and saves as parquet.
"""
import pandas as pd
import numpy as np
import os
import sys
import io
import time
import re
import unicodedata
from multiprocessing import Pool, cpu_count

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import (
    LEGAL_SUFFIXES, INDIAN_STATES_TRANSLITERATION,
    normalize_unicode, transliterate_indic, replace_indian_states,
    extract_numbers
)

BASE = r"f:\TECH\ML challenge\student_resource\dataset"
OUTPUT = r"f:\TECH\ML challenge\student_resource\preprocessed"
os.makedirs(OUTPUT, exist_ok=True)


def fast_clean_text(series: pd.Series) -> pd.Series:
    """Vectorized text cleaning on a pandas Series."""
    s = series.fillna('').astype(str)
    s = s.str.lower().str.strip()
    s = s.replace({'nan': '', 'null': ''})
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace('<<', '', regex=False).str.replace('>>', '', regex=False)
    s = s.str.replace(r'[,;\:\-_/]+', ' ', regex=True)
    s = s.str.replace(r'[(){}\[\]"\']', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True)
    s = s.str.strip(' .')
    return s


def fast_standardize_suffix(name: str) -> str:
    """Standardize legal suffixes in a business name."""
    if not name:
        return ''
    tokens = name.split()
    result = []
    for token in tokens:
        lookup = token.strip('.,;:()[]').replace('.', '')
        if lookup in LEGAL_SUFFIXES:
            result.append(LEGAL_SUFFIXES[lookup])
        else:
            result.append(token)
    return ' '.join(result)


def fast_extract_core(name: str) -> str:
    """Extract core business name without stopwords/suffixes."""
    if not name:
        return ''
    stopwords = {'the', 'of', 'and', 'for', 'in', 'at', 'by', 'a', 'an', 'to', 'on',
                 'dba', 'llc', 'llp', 'incorporated', 'corporation', 'limited',
                 'private', 'company', 'sarl', 'sas', 'sasu', 'sa', 'sci', 'eurl',
                 'snc', 'groupe', 'fils', 'et'}
    tokens = name.split()
    core = []
    for t in tokens:
        cleaned = re.sub(r'[^a-z0-9\u0900-\u0DFF]', '', t)
        if cleaned and cleaned not in stopwords and len(cleaned) > 1:
            core.append(cleaned)
    return ' '.join(core) if core else name


def process_single_text(text: str) -> str:
    """Process a single text: replace states → transliterate → normalize unicode → lowercase."""
    if not text:
        return ''
    text = replace_indian_states(text)
    text = transliterate_indic(text)
    # Re-clean after transliteration
    text = text.lower().strip()
    text = re.sub(r'[,;\:\-_/]+', ' ', text)
    text = re.sub(r'[(){}\[\]"\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Unicode normalize (remove accents)
    text = normalize_unicode(text).lower()
    return text


def process_chunk(chunk_data):
    """Process a chunk of (names, addresses) — for multiprocessing."""
    names, addresses, chunk_idx = chunk_data
    
    name_normalized = []
    name_core = []
    addr_normalized = []
    addr_numbers = []
    
    for name, addr in zip(names, addresses):
        # Process name
        n = process_single_text(name)
        n_std = fast_standardize_suffix(n)
        n_core = fast_extract_core(n_std)
        name_normalized.append(n_std)
        name_core.append(n_core)
        
        # Process address
        a = process_single_text(addr)
        addr_normalized.append(a)
        addr_numbers.append(' '.join(re.findall(r'\d+', a)) if a else '')
    
    return chunk_idx, name_normalized, name_core, addr_normalized, addr_numbers


def preprocess_dataframe_fast(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """
    Fast preprocessing using vectorized ops for cleaning + multiprocessing for transliteration.
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    
    total = len(df)
    print(f"  Preprocessing {total:,} records with {n_workers} workers...")
    
    # Step 1: Fast vectorized cleaning (very fast)
    t0 = time.time()
    name_clean = fast_clean_text(df['business_name'])
    addr_clean = fast_clean_text(df['business_address'])
    print(f"    Step 1 (vectorized cleaning): {time.time()-t0:.1f}s")
    
    # Step 2: Transliteration + normalization (needs per-row, use multiprocessing)
    t1 = time.time()
    
    names_list = name_clean.tolist()
    addrs_list = addr_clean.tolist()
    
    # Split into chunks for multiprocessing
    chunk_size = max(len(names_list) // n_workers, 1)
    chunks = []
    for i in range(0, len(names_list), chunk_size):
        end = min(i + chunk_size, len(names_list))
        chunks.append((names_list[i:end], addrs_list[i:end], i))
    
    # Process chunks
    all_name_normalized = [''] * total
    all_name_core = [''] * total
    all_addr_normalized = [''] * total
    all_addr_numbers = [''] * total
    
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(process_chunk, chunks)
    else:
        results = [process_chunk(c) for c in chunks]
    
    # Reassemble results
    for chunk_idx, name_norm, name_c, addr_norm, addr_nums in results:
        size = len(name_norm)
        all_name_normalized[chunk_idx:chunk_idx+size] = name_norm
        all_name_core[chunk_idx:chunk_idx+size] = name_c
        all_addr_normalized[chunk_idx:chunk_idx+size] = addr_norm
        all_addr_numbers[chunk_idx:chunk_idx+size] = addr_nums
    
    print(f"    Step 2 (transliteration+normalization): {time.time()-t1:.1f}s")
    
    # Build result dataframe
    result = df.copy()
    result['name_clean'] = name_clean.values
    result['name_normalized'] = all_name_normalized
    result['name_core'] = all_name_core
    result['addr_clean'] = addr_clean.values
    result['addr_normalized'] = all_addr_normalized
    result['addr_numbers'] = all_addr_numbers
    
    return result


if __name__ == '__main__':
    FILES = [
        ("train", "train_source1.tsv", "train_s1"),
        ("train", "train_source2.tsv", "train_s2"),
        ("train", "train_source3.tsv", "train_s3"),
        ("test", "test_source1.tsv", "test_s1"),
        ("test", "test_source2.tsv", "test_s2"),
        ("test", "test_source3.tsv", "test_s3"),
    ]
    
    total_start = time.time()
    
    for split, filename, output_name in FILES:
        filepath = os.path.join(BASE, split, filename)
        outpath = os.path.join(OUTPUT, f"{output_name}.parquet")
        
        # Skip if already processed
        if os.path.exists(outpath):
            size_mb = os.path.getsize(outpath) / 1024 / 1024
            print(f"\n  SKIP {output_name} (already exists, {size_mb:.0f} MB)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        
        start = time.time()
        
        # Load
        print(f"  Loading...")
        df = pd.read_csv(filepath, sep="\t")
        load_time = time.time() - start
        print(f"  Loaded {len(df):,} rows in {load_time:.1f}s")
        
        # Preprocess (fast)
        df_processed = preprocess_dataframe_fast(df)
        
        # Save
        df_processed.to_parquet(outpath, index=False)
        elapsed = time.time() - start
        print(f"  Saved → {outpath}")
        print(f"  Total: {elapsed:.1f}s ({len(df)/elapsed:.0f} rows/sec)")
    
    # Ground truth
    print(f"\n{'='*70}")
    print("Copying ground truth...")
    gt = pd.read_csv(os.path.join(BASE, "train", "train_ground_truth.tsv"), sep="\t")
    gt.to_parquet(os.path.join(OUTPUT, "train_gt.parquet"), index=False)
    print(f"  Saved {len(gt):,} rows")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ALL DONE! Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")
