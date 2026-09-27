import pandas as pd
import os

def generate_submission(final_matches: pd.DataFrame, candidates_df: pd.DataFrame, output_dir: str):
    """
    Takes the final matches and the blocking candidates and formats them 
    according to the challenge submission requirements.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. format matching_results.tsv
    matching = final_matches.groupby('s1_entity_id')['S2_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    matching.columns = ['source1_entity_id', 'matched_entity_ids']
    match_path = os.path.join(output_dir, "matching_results.tsv")
    matching.to_csv(match_path, sep='\t', index=False)
    print(f"Final matches saved to {match_path}")
    
    # 2. format candidate_pairs.tsv
    candidates = candidates_df.groupby('s1_entity_id')['S2_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    candidates.columns = ['source1_entity_id', 'candidate_entity_ids']
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    candidates.to_csv(cand_path, sep='\t', index=False)
    print(f"Candidate pairs saved to {cand_path}")
    
    return matching, candidates
