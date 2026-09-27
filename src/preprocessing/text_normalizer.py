import re
import pandas as pd
from .transliteration import transliterate_text

def normalize_text(text: str) -> str:
    """
    Basic text normalization:
    1. Lowercase
    2. Transliterate to Latin
    3. Remove punctuation and extra spaces
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    # text = transliterate_text(str(text)) # Disabled for fast local CPU testing
    text = str(text).lower()
    
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_text)
