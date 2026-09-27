import re
import pandas as pd

def extract_pincode(address: str) -> str:
    """Extracts PIN/ZIP code from address string"""
    if not address:
        return ""
    # Look for 5-digit (US) or 6-digit (India) numbers, often at the end
    match = re.search(r'\b(\d{5,6})\b', str(address))
    if match:
        return match.group(1)
    return ""

def extract_numbers(address: str) -> str:
    """Extracts all numbers (house numbers, street numbers) from address"""
    if not address:
        return ""
    numbers = re.findall(r'\b\d+\b', str(address))
    return " ".join(numbers)

def extract_street_type(address: str) -> str:
    """Extracts common street types to match across abbreviations"""
    if not address:
        return ""
    types = []
    if re.search(r'\b(st|street)\b', address): types.append('street')
    if re.search(r'\b(rd|road)\b', address): types.append('road')
    if re.search(r'\b(ave|avenue)\b', address): types.append('avenue')
    if re.search(r'\b(blvd|boulevard)\b', address): types.append('boulevard')
    if re.search(r'\b(dr|drive)\b', address): types.append('drive')
    if re.search(r'\b(ln|lane)\b', address): types.append('lane')
    if re.search(r'\b(ct|court)\b', address): types.append('court')
    if re.search(r'\b(pl|place)\b', address): types.append('place')
    if re.search(r'\b(sq|square)\b', address): types.append('square')
    if re.search(r'\b(hwy|highway)\b', address): types.append('highway')
    return " ".join(types)

def extract_state_code(address: str) -> str:
    """Extracts common state codes (very naive, can be expanded)"""
    if not address:
        return ""
    # Only 2 letter uppercase words bounded by spaces/punctuation (done before lowercasing)
    # Actually, we assume input is already normalized (lowercased)
    # We will skip this for now or rely on TF-IDF word matches
    return ""

def parse_address_regex(address: str) -> dict:
    """
    Regex fallback for parsing addresses since libpostal might be too slow.
    Assumes address is already normalized (lowercased, transliterated, punctuation removed).
    """
    if pd.isna(address) or str(address).strip() == "":
        return {
            'pincode': '',
            'numbers': '',
            'street_type': ''
        }
    
    return {
        'pincode': extract_pincode(address),
        'numbers': extract_numbers(address),
        'street_type': extract_street_type(address)
    }

def apply_address_parsing(df: pd.DataFrame, address_col: str = 'norm_address') -> pd.DataFrame:
    """Applies regex address parsing and adds extracted components as new columns"""
    parsed = df[address_col].apply(parse_address_regex)
    df_parsed = pd.DataFrame(parsed.tolist(), index=df.index)
    
    # Prefix columns
    df_parsed = df_parsed.add_prefix('addr_')
    return pd.concat([df, df_parsed], axis=1)
