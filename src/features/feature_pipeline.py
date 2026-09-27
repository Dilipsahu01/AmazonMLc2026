import pandas as pd
from .name_features import compute_name_features
from .address_features import compute_address_features
from .phonetic_features import compute_phonetic_features
from .cross_features import compute_cross_features

def build_features(df_pairs: pd.DataFrame, df_s1: pd.DataFrame, df_s2: pd.DataFrame, s2_prefix: str = 'S2') -> pd.DataFrame:
    """
    Given a dataframe of candidate pairs (s1_entity_id, S2_entity_id, blocking_score),
    joins the original text data and computes all ML features.
    """
    print("Building features for candidate pairs...")
    
    # Only keep the necessary columns to save RAM (drop raw text)
    cols_to_keep = ['entity_id', 'norm_name', 'norm_address', 'addr_pincode', 'addr_numbers', 'addr_street_type']
    
    df_s1_slim = df_s1[[c for c in cols_to_keep if c in df_s1.columns]]
    df_s2_slim = df_s2[[c for c in cols_to_keep if c in df_s2.columns]]
    
    # 1. Join S1 data
    df_s1_renamed = df_s1_slim.add_prefix('s1_')
    merged = df_pairs.merge(df_s1_renamed, on='s1_entity_id', how='left')
    
    # 2. Join S2 data
    df_s2_renamed = df_s2_slim.add_prefix(f'{s2_prefix}_')
    merged = merged.merge(df_s2_renamed, on=f'{s2_prefix}_entity_id', how='left')
    
    # 3. Compute Features
    print("  Computing name features...")
    name_feats = compute_name_features(merged, prefix1='s1', prefix2=s2_prefix)
    
    print("  Computing address features...")
    addr_feats = compute_address_features(merged, prefix1='s1', prefix2=s2_prefix)
    
    print("  Computing phonetic features...")
    phonetic_feats = compute_phonetic_features(merged, prefix1='s1', prefix2=s2_prefix)
    
    print("  Computing cross-field features...")
    cross_feats = compute_cross_features(merged, prefix1='s1', prefix2=s2_prefix)
    
    # 4. Combine all features
    # Start with the blocking score as the first feature
    feature_matrix = pd.DataFrame({'blocking_score': merged['blocking_score']})
    feature_matrix = pd.concat([
        feature_matrix, 
        name_feats, 
        addr_feats,
        phonetic_feats,
        cross_feats
    ], axis=1)
    
    return feature_matrix
