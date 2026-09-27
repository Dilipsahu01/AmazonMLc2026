import re
import pandas as pd

# Common business abbreviations mapping
ABBREVIATIONS = {
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\binc\b': 'incorporated',
    r'\bcorp\b': 'corporation',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\b&\b': 'and',
}

def expand_abbreviations(text: str) -> str:
    """Expands common business abbreviations for standardisation."""
    if not text:
        return ""
    
    # Assumes text is already lowercased and transliterated
    for abbr, expanded in ABBREVIATIONS.items():
        text = re.sub(abbr, expanded, text)
        
    # Standardize 'dba' (doing business as)
    text = re.sub(r'\bdba\b', 'doing business as', text)
    
    # Remove website domain suffixes like .com, .in (from our EDA, we saw www.shivshakti.com)
    text = re.sub(r'\bwww\.\S+', '', text)
    text = re.sub(r'\S+\.com\b', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_name(text: str) -> str:
    """Pipeline for name normalization"""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    # Expand abbreviations
    text = expand_abbreviations(str(text))
    return text

def normalize_name_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_name)
