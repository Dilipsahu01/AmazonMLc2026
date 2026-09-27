import pandas as pd
from thefuzz import fuzz
import jellyfish

def compute_address_features(df: pd.DataFrame, prefix1='s1', prefix2='S2') -> pd.DataFrame:
    """
    Computes similarity features between S1 and S2 address components.
    df must contain columns:
        <prefix1>_norm_address, <prefix2>_norm_address
        <prefix1>_addr_pincode, <prefix2>_addr_pincode
        <prefix1>_addr_numbers, <prefix2>_addr_numbers
        <prefix1>_addr_street_type, <prefix2>_addr_street_type
    """
    features = pd.DataFrame()
    
    # Full Address Similarity
    add1 = df[f"{prefix1}_norm_address"].fillna("").astype(str).tolist()
    add2 = df[f"{prefix2}_norm_address"].fillna("").astype(str).tolist()
    
    f_addr_jw = []
    f_addr_tokenset = []
    
    for a1, a2 in zip(add1, add2):
        if not a1 and not a2:
            f_addr_jw.append(-1.0)
            f_addr_tokenset.append(-1)
            continue
        if not a1 or not a2:
            f_addr_jw.append(0.0)
            f_addr_tokenset.append(0)
            continue
            
        f_addr_jw.append(jellyfish.jaro_winkler_similarity(a1, a2))
        f_addr_tokenset.append(fuzz.token_set_ratio(a1, a2))
        
    features['addr_jaro_winkler'] = f_addr_jw
    features['addr_token_set_ratio'] = f_addr_tokenset
    
    # Exact Match Features (Pincode, Numbers, Street Type)
    for comp in ['pincode', 'numbers', 'street_type']:
        c1 = df[f"{prefix1}_addr_{comp}"].fillna("").astype(str)
        c2 = df[f"{prefix2}_addr_{comp}"].fillna("").astype(str)
        
        # 1 if exact match, 0 if mismatch, -1 if either is missing
        exact_match = (c1 == c2).astype(int)
        missing_mask = (c1 == "") | (c2 == "")
        exact_match[missing_mask] = -1
        
        features[f'{comp}_exact_match'] = exact_match.values
        
    return features
