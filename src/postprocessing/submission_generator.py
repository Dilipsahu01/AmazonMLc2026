import pandas as pd
import os

def generate_submission(final_matches: pd.DataFrame, output_path: str):
    """
    Takes the final matches dataframe containing ['s1_entity_id', 'S2_entity_id'] 
    and formats it according to the submission requirements.
    
    Output Format:
    source1_entity_id \t matched_entity_ids
    S1-xxx \t S2-yyy,S3-zzz
    """
    # Group by s1_entity_id and join the S2/S3 entity ids with commas
    submission = final_matches.groupby('s1_entity_id')['S2_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    submission.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Save to TSV
    submission.to_csv(output_path, sep='\t', index=False)
    print(f"Submission saved to {output_path}")
    return submission
