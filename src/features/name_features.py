import pandas as pd
from thefuzz import fuzz
import jellyfish

def compute_name_features(df: pd.DataFrame, prefix1='s1', prefix2='S2') -> pd.DataFrame:
    """
    Computes name similarity features between S1 and S2 candidates.
    df must contain columns: <prefix1>_norm_name, <prefix2>_norm_name
    """
    col1 = f"{prefix1}_norm_name"
    col2 = f"{prefix2}_norm_name"
    
    # We use a vectorized approach or list comprehension for speed
    names1 = df[col1].fillna("").astype(str).tolist()
    names2 = df[col2].fillna("").astype(str).tolist()
    
    # Initialize feature lists
    f_jw = []
    f_lev = []
    f_token_sort = []
    f_token_set = []
    f_len_diff = []
    
    for n1, n2 in zip(names1, names2):
        if not n1 and not n2:
            f_jw.append(1.0)
            f_lev.append(1.0)
            f_token_sort.append(100)
            f_token_set.append(100)
            f_len_diff.append(0)
            continue
            
        f_jw.append(jellyfish.jaro_winkler_similarity(n1, n2))
        
        # Levenshtein distance normalized by max length
        max_len = max(len(n1), len(n2))
        dist = jellyfish.levenshtein_distance(n1, n2)
        f_lev.append(1.0 - (dist / max_len) if max_len > 0 else 0)
        
        f_token_sort.append(fuzz.token_sort_ratio(n1, n2))
        f_token_set.append(fuzz.token_set_ratio(n1, n2))
        
        f_len_diff.append(abs(len(n1) - len(n2)))
        
    features = pd.DataFrame({
        'name_jaro_winkler': f_jw,
        'name_norm_levenshtein': f_lev,
        'name_token_sort_ratio': f_token_sort,
        'name_token_set_ratio': f_token_set,
        'name_len_diff': f_len_diff
    })
    
    return features
