import pandas as pd
import numpy as np
import os
import gc
from src.data_loader import load_and_preprocess
from src.blocking.pipeline import BlockingPipeline
from src.features.feature_pipeline import build_features
from src.matching.lgbm_matcher import LGBMMatcher
from src.matching.singleton_detector import apply_singleton_threshold
from src.postprocessing.submission_generator import generate_submission

def run_end_to_end_pipeline(data_dir: str, output_dir: str, sample_size: int = None):
    """
    Runs the entire Amazon ML Challenge pipeline properly split into Train and Test.
    """
    print("=== AMAZON ML CHALLENGE ER PIPELINE ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # PHASE A: TRAINING
    # =========================================================================
    print("\n--- PHASE A: TRAINING ---")
    print("[1] Loading Training Data (Subsampled for RAM constraints)...")
    
    def get_path(filename, subdir):
        flat_path = f"{data_dir}/{filename}"
        if os.path.exists(flat_path):
            return flat_path
        return f"{data_dir}/{subdir}/{filename}"

    # For training, we must sample to fit in memory
    train_sample_size = sample_size if sample_size else 50000 
    df_s1_train = load_and_preprocess(get_path("train_source1.tsv", "train")).head(train_sample_size)
    
    # Load full S2 and S3 for blocking
    df_s2_train = load_and_preprocess(get_path("train_source2.tsv", "train"))
    df_s3_train = load_and_preprocess(get_path("train_source3.tsv", "train"))
    df_gt = pd.read_csv(get_path("train_ground_truth.tsv", "train"), sep='\t', na_filter=False)
    
    print("\n[2] Blocking Training Candidates...")
    blocker = BlockingPipeline(top_k=20, ngram_range=(3, 3))
    
    cand_s2_train = blocker.run(df_s1_train, df_s2_train, s2_prefix='S2')
    cand_s3_train = blocker.run(df_s1_train, df_s3_train, s2_prefix='S3')
    
    # Combine S2 and S3 candidates
    cand_s3_train.rename(columns={'S3_entity_id': 'S2_entity_id'}, inplace=True)
    candidates_train = pd.concat([cand_s2_train, cand_s3_train], ignore_index=True)
    
    print("\n[3] Generating Training Labels...")
    true_pairs = set()
    s1_ids = set(df_s1_train['entity_id'])
    
    df_gt_sample = df_gt[df_gt['source1_entity_id'].isin(s1_ids)]
    for _, row in df_gt_sample.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if matches_str:
            for match in str(matches_str).split(','):
                true_pairs.add((s1_id, match))
                
    candidates_train['is_match'] = candidates_train.apply(
        lambda r: 1 if (r['s1_entity_id'], r['S2_entity_id']) in true_pairs else 0, axis=1
    )
    
    print("\n[4] Extracting Train Features...")
    df_s2_s3_combined = pd.concat([df_s2_train, df_s3_train], ignore_index=True)
    X_train = build_features(candidates_train, df_s1_train, df_s2_s3_combined, s2_prefix='S2')
    y_train = candidates_train['is_match'].values
    
    print("\n[5] Training LightGBM Matcher...")
    matcher = LGBMMatcher()
    matcher.fit(X_train, y_train)
    
    # Clear RAM
    del df_s1_train, df_s2_train, df_s3_train, candidates_train, X_train, y_train, df_s2_s3_combined
    gc.collect()
    
    # =========================================================================
    # PHASE B: INFERENCE (TEST SET)
    # =========================================================================
    print("\n--- PHASE B: INFERENCE (TEST) ---")
    print("[1] Loading Test Data...")
    df_s1_test = load_and_preprocess(get_path("test_source1.tsv", "test"))
    df_s2_test = load_and_preprocess(get_path("test_source2.tsv", "test"))
    df_s3_test = load_and_preprocess(get_path("test_source3.tsv", "test"))
    
    if sample_size:
        df_s1_test = df_s1_test.head(sample_size).copy()
        
    s1_universe = df_s1_test['entity_id'].tolist()
    
    print("\n[2] Blocking Test Candidates...")
    cand_s2_test = blocker.run(df_s1_test, df_s2_test, s2_prefix='S2')
    cand_s3_test = blocker.run(df_s1_test, df_s3_test, s2_prefix='S3')
    
    cand_s3_test.rename(columns={'S3_entity_id': 'S2_entity_id'}, inplace=True)
    candidates_test = pd.concat([cand_s2_test, cand_s3_test], ignore_index=True)
    
    print("\n[3] Extracting Test Features...")
    df_s2_s3_test = pd.concat([df_s2_test, df_s3_test], ignore_index=True)
    X_test = build_features(candidates_test, df_s1_test, df_s2_s3_test, s2_prefix='S2')
    
    print("\n[4] Predicting & Applying Thresholds...")
    candidates_test['match_prob'] = matcher.predict_proba(X_test)
    final_matches = apply_singleton_threshold(candidates_test, singleton_threshold=0.6, match_threshold=0.4)
    
    print("\n[5] Generating Submission Files...")
    matching, candidates = generate_submission(final_matches, candidates_test, s1_universe, output_dir)
    
    print("Pipeline Execution Complete!")
    return matching
