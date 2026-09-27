import pandas as pd

def get_country_partitions(df: pd.DataFrame, country_col: str = 'country') -> dict:
    """
    Groups the dataframe by country to create independent blocking partitions.
    Returns a dictionary mapping country names to boolean masks or dataframes.
    For extreme scale, returning indices or masks is more memory efficient.
    """
    partitions = {}
    for country in df[country_col].unique():
        # Keep track of indices where country matches
        partitions[country] = df[country_col] == country
        
    return partitions

def filter_by_country(df: pd.DataFrame, country: str, country_col: str = 'country') -> pd.DataFrame:
    """Returns subset of df for a specific country."""
    return df[df[country_col] == country].copy()
