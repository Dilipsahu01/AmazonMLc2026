import pandas as pd
import jellyfish
from thefuzz import fuzz

def compute_phonetic_features(df: pd.DataFrame, prefix1='s1', prefix2='S2') -> pd.DataFrame:
    """
    Computes phonetic similarity features (Soundex and Match Rating Approach).
    Useful for catching names that sound identical but are spelled very differently.
    """
    col1 = f"{prefix1}_norm_name"
    col2 = f"{prefix2}_norm_name"
    
    names1 = df[col1].fillna("").astype(str).tolist()
    names2 = df[col2].fillna("").astype(str).tolist()
    
    f_soundex_match = []
    f_mra_match = []
    
    for n1, n2 in zip(names1, names2):
        if not n1 and not n2:
            f_soundex_match.append(1)
            f_mra_match.append(1)
            continue
            
        # Get first word of the name for phonetic comparison (usually the defining noun)
        w1 = n1.split()[0] if n1 else ""
        w2 = n2.split()[0] if n2 else ""
        
        # Jellyfish strictly requires alphabetical characters for phonetic algorithms
        alpha_w1 = ''.join(c for c in w1 if c.isalpha()) if w1 else ""
        alpha_w2 = ''.join(c for c in w2 if c.isalpha()) if w2 else ""
        
        # Soundex
        s1 = ""
        s2 = ""
        try:
            s1 = jellyfish.soundex(alpha_w1) if alpha_w1 else ""
            s2 = jellyfish.soundex(alpha_w2) if alpha_w2 else ""
        except ValueError:
            pass
            
        f_soundex_match.append(1 if (s1 and s1 == s2) else 0)
        
        # Match Rating Approach (MRA)
        m1 = ""
        m2 = ""
        try:
            m1 = jellyfish.match_rating_codex(alpha_w1) if alpha_w1 else ""
            m2 = jellyfish.match_rating_codex(alpha_w2) if alpha_w2 else ""
        except ValueError:
            pass
        
        # Fuzzy match on the MRA code
        f_mra_match.append(fuzz.ratio(m1, m2))
        
    features = pd.DataFrame({
        'phonetic_soundex_match': f_soundex_match,
        'phonetic_mra_ratio': f_mra_match
    })
    
    return features
