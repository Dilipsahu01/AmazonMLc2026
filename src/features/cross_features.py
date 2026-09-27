import pandas as pd
import jellyfish
from thefuzz import fuzz

def compute_cross_features(df: pd.DataFrame, prefix1='s1', prefix2='S2') -> pd.DataFrame:
    """
    Computes cross-field features.
    Sometimes a Name is accidentally placed in the Address field (or vice versa).
    We check the similarity of S1 Name vs S2 Address, and S1 Address vs S2 Name.
    """
    name1 = df[f"{prefix1}_norm_name"].fillna("").astype(str).tolist()
    addr1 = df[f"{prefix1}_norm_address"].fillna("").astype(str).tolist()
    
    name2 = df[f"{prefix2}_norm_name"].fillna("").astype(str).tolist()
    addr2 = df[f"{prefix2}_norm_address"].fillna("").astype(str).tolist()
    
    f_n1_a2 = []
    f_a1_n2 = []
    
    for n1, a1, n2, a2 in zip(name1, addr1, name2, addr2):
        # S1 Name vs S2 Address
        if n1 and a2:
            f_n1_a2.append(fuzz.token_set_ratio(n1, a2))
        else:
            f_n1_a2.append(0)
            
        # S1 Address vs S2 Name
        if a1 and n2:
            f_a1_n2.append(fuzz.token_set_ratio(a1, n2))
        else:
            f_a1_n2.append(0)
            
    features = pd.DataFrame({
        'cross_name1_addr2_tokenset': f_n1_a2,
        'cross_addr1_name2_tokenset': f_a1_n2
    })
    
    return features
