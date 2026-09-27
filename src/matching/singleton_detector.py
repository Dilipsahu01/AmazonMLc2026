import pandas as pd

def apply_singleton_threshold(predictions_df: pd.DataFrame, singleton_threshold=0.6, match_threshold=0.4) -> pd.DataFrame:
    """
    Applies the singleton-aware decision logic.
    For each query in S1:
    1. If the BEST match probability is < singleton_threshold, we return EMPTY (singleton).
    2. Otherwise, we return ALL matches with probability >= match_threshold.
    
    predictions_df must contain: ['s1_entity_id', 'S2_entity_id', 'match_prob']
    Returns a dataframe with the final selected matches.
    """
    # 1. Find the best match probability for each S1 entity
    best_probs = predictions_df.groupby('s1_entity_id')['match_prob'].max().reset_index()
    best_probs.rename(columns={'match_prob': 'best_prob'}, inplace=True)
    
    # 2. Join the best probability back to the predictions
    merged = predictions_df.merge(best_probs, on='s1_entity_id', how='left')
    
    # 3. Apply logic
    # Keep only rows where best_prob >= singleton_threshold AND match_prob >= match_threshold
    mask = (merged['best_prob'] >= singleton_threshold) & (merged['match_prob'] >= match_threshold)
    
    final_matches = merged[mask].copy()
    
    # Drop the temporary column
    final_matches.drop(columns=['best_prob'], inplace=True)
    
    return final_matches
