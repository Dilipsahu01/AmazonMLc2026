import pandas as pd
import numpy as np
import time
from . import config
from .preprocessing import (
    normalize_series,
    normalize_name_series,
    normalize_country_series,
    apply_address_parsing
)

def load_and_preprocess(filepath, is_ground_truth=False):
    """
    Loads a TSV file and applies the full preprocessing pipeline.
    """
    print(f"Loading {filepath}...")
    t0 = time.time()
    df = pd.read_csv(filepath, sep='\t', na_filter=False, dtype=str)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    
    if is_ground_truth:
        return df
    
    # 1. Normalize Country
    print("  Normalizing country...")
    t0 = time.time()
    df['country'] = normalize_country_series(df['country'])
    
    # 2. Normalize and Transliterate Name
    print("  Normalizing name (transliteration + lowercasing)...")
    t0 = time.time()
    df['norm_name'] = normalize_series(df['business_name'])
    df['norm_name'] = normalize_name_series(df['norm_name'])
    
    # 3. Normalize and Transliterate Address
    print("  Normalizing address (transliteration + lowercasing)...")
    t0 = time.time()
    df['norm_address'] = normalize_series(df['business_address'])
    
    # 4. Extract Address Components (Regex)
    print("  Extracting address components...")
    t0 = time.time()
    df = apply_address_parsing(df, address_col='norm_address')
    
    print(f"  Preprocessing complete. Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

def test_pipeline():
    """Quick test on a tiny sample"""
    sample_data = {
        'entity_id': ['S1-1', 'S2-2', 'S3-3'],
        'business_name': ['Dahlia Power Reliable Scientific LLC', 'राम मार्केटिंग प्राइवेट लिमिटेड', 'www.shivshakti.com'],
        'business_address': ['630 45th Terrace, Kansas City, MO', 'KH NO. -570/13, NEW DELHI', ''],
        'country': ['US', 'India', 'India']
    }
    df = pd.DataFrame(sample_data)
    
    df['country'] = normalize_country_series(df['country'])
    df['norm_name'] = normalize_series(df['business_name'])
    df['norm_name'] = normalize_name_series(df['norm_name'])
    df['norm_address'] = normalize_series(df['business_address'])
    df = apply_address_parsing(df, address_col='norm_address')
    
    print("Pipeline Test Results:")
    for _, row in df.iterrows():
        print(f"\nOriginal: {row['business_name']} | {row['business_address']}")
        print(f"Name:     {row['norm_name']}")
        print(f"Address:  {row['norm_address']}")
        print(f"Parsed:   PIN={row['addr_pincode']} NUMS={row['addr_numbers']} TYPE={row['addr_street_type']}")

if __name__ == "__main__":
    test_pipeline()
