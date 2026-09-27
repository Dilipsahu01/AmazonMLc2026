import pandas as pd

def normalize_country(country: str) -> str:
    """
    Standardize the country field.
    The problem statement specifies it must remain an open-set string filter.
    France is explicitly mentioned as being in the test set.
    We just clean whitespace and lowercase it for reliable matching.
    """
    if pd.isna(country) or str(country).strip() == "":
        return "unknown"
    return str(country).strip().lower()

def normalize_country_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_country)
