import pandas as pd
import numpy as np
import os
import gc
from src.data_loader import load_and_preprocess
from src.blocking.pipeline import BlockingPipeline
from src.features.feature_pipeline import build_features
from src.matching.lgbm_matcher import LGBMMatcher
from src.matching.singleton_detector import apply_singleton_threshold

def run_end_to_end_pipeline(data_dir: str, output_dir: str, sample_size: int = None):
    """
    Runs the entire Amazon ML Challenge pipeline.
    If sample_size is provided, it runs on a subset (for quick testing).
    """
    print("=== AMAZON ML CHALLENGE ER PIPELINE ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD AND PREPROCESS DATA
    # ---------------------------------------------------------
    print("\n[1] Loading and Preprocessing Training Data...")
    df_s1 = load_and_preprocess(f"{data_dir}/train/train_source1.tsv")
    df_s2 = load_and_preprocess(f"{data_dir}/train/train_source2.tsv")
    
    if sample_size:
        df_s1 = df_s1.head(sample_size).copy()
        
    df_gt = pd.read_csv(f"{data_dir}/train/train_ground_truth.tsv", sep='\t', na_filter=False)
    
    # ---------------------------------------------------------
    # 2. BLOCKING (Candidate Generation)
    # ---------------------------------------------------------
    print("\n[2] Running TF-IDF Blocking...")
    blocker = BlockingPipeline(top_k=50, ngram_range=(3, 3))
    
    # Filter S2 to only valid countries to save RAM during blocking
    valid_countries = df_s1['country'].unique()
    df_s2_filtered = df_s2[df_s2['country'].isin(valid_countries)]
    
    candidates_df = blocker.run(df_s1, df_s2_filtered, s2_prefix='S2')
    
    # ---------------------------------------------------------
    # 3. LABELING THE CANDIDATES (For Training)
    # ---------------------------------------------------------
    print("\n[3] Generating Training Labels...")
    # Parse ground truth to a fast lookup set
    true_pairs = set()
    s1_ids = set(df_s1['entity_id'])
    
    df_gt_sample = df_gt[df_gt['source1_entity_id'].isin(s1_ids)]
    for _, row in df_gt_sample.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if matches_str:
            matches = str(matches_str).split(',')
            for match in matches:
                true_pairs.add((s1_id, match))
                
    # Assign labels
    candidates_df['is_match'] = candidates_df.apply(
        lambda r: 1 if (r['s1_entity_id'], r['S2_entity_id']) in true_pairs else 0, 
        axis=1
    )
    
    print(f"Total candidates: {len(candidates_df)}")
    print(f"True matches in candidates: {candidates_df['is_match'].sum()}")
    
    # ---------------------------------------------------------
    # 4. FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\n[4] Extracting Machine Learning Features...")
    # This returns just the feature columns
    X_train = build_features(candidates_df, df_s1, df_s2, s2_prefix='S2')
    y_train = candidates_df['is_match'].values
    
    # ---------------------------------------------------------
    # 5. MODEL TRAINING (LightGBM)
    # ---------------------------------------------------------
    print("\n[5] Training LightGBM Matcher...")
    matcher = LGBMMatcher()
    # In a real scenario, we'd split a validation set. For this pipeline we train on all.
    matcher.fit(X_train, y_train)
    
    importances = matcher.get_feature_importances()
    print("\nTop 5 Important Features:")
    print(importances.head(5))
    
    # ---------------------------------------------------------
    # 6. INFERENCE & POST-PROCESSING
    # ---------------------------------------------------------
    print("\n[6] Predicting & Applying Singleton Thresholds...")
    candidates_df['match_prob'] = matcher.predict_proba(X_train)
    
    # Apply the threshold logic
    # singleton_threshold = 0.6: If best match < 0.6, return EMPTY
    # match_threshold = 0.4: Return all matches >= 0.4
    final_matches = apply_singleton_threshold(candidates_df, singleton_threshold=0.6, match_threshold=0.4)
    
    print(f"Final predicted matches: {len(final_matches)}")
    
    # ---------------------------------------------------------
    # 7. GENERATE SUBMISSION
    # ---------------------------------------------------------
    print("\n[7] Generating Submission File...")
    from src.postprocessing.submission_generator import generate_submission
    sub_path = os.path.join(output_dir, "submission.tsv")
    generate_submission(final_matches, sub_path)
    
    print("Pipeline Execution Complete!")
    return final_matches
