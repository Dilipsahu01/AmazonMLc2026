import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
from src.data_loader import load_and_preprocess
from src.blocking.union_blocker import UnionBlocker
from src.features.feature_pipeline import build_features
from src.matching.lgbm_matcher import LGBMMatcher

def evaluate_pipeline(sample_size=20000):
    data_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    print("=== PIPELINE EVALUATION (LOCAL) ===")
    print(f"Loading {sample_size} S1 records for validation...")
    
    df_s1 = load_and_preprocess(f"{data_dir}/train_source1.tsv").head(sample_size)
    df_s2 = load_and_preprocess(f"{data_dir}/train_source2.tsv").head(10000)
    df_s3 = load_and_preprocess(f"{data_dir}/train_source3.tsv").head(10000)
    df_gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep='\t', na_filter=False)
    
    from src.blocking.pipeline import BlockingPipeline
    print("\n[1] Blocking (TF-IDF only for fast local evaluation)...")
    blocker = BlockingPipeline(top_k=20, ngram_range=(3, 3))
    cand_s2 = blocker.run(df_s1, df_s2, s2_prefix='S2')
    cand_s3 = blocker.run(df_s1, df_s3, s2_prefix='S3')
    
    cand_s3.rename(columns={'S3_entity_id': 'S2_entity_id'}, inplace=True)
    candidates = pd.concat([cand_s2, cand_s3], ignore_index=True)
    
    print("\n[2] Generating True Labels for Candidates...")
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
                
    y = np.array([1 if (r['s1_entity_id'], r['S2_entity_id']) in true_pairs else 0 
                  for _, r in candidates.iterrows()])
    
    found = sum(y)
    total_true = len(true_pairs)
    blocking_recall = found / total_true if total_true > 0 else 0
    print(f"-> Blocking Recall @ Top 20: {blocking_recall:.4f}")
    
    print("\n[3] Extracting Features...")
    df_s2_s3 = pd.concat([df_s2, df_s3], ignore_index=True)
    X = build_features(candidates, df_s1, df_s2_s3, s2_prefix='S2')
    
    print("\n[4] Training Models (LightGBM vs CatBoost)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. LightGBM
    matcher_lgb = LGBMMatcher()
    matcher_lgb.fit(X_train, y_train)
    y_pred_prob_lgb = matcher_lgb.predict_proba(X_val)
    y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)
    f05_lgb = fbeta_score(y_val, y_pred_lgb, beta=0.5, zero_division=0)
    
    # 2. CatBoost
    from src.matching.catboost_matcher import CatBoostMatcher
    matcher_cb = CatBoostMatcher()
    matcher_cb.fit(X_train, y_train)
    y_pred_prob_cb = matcher_cb.predict_proba(X_val)
    y_pred_cb = (y_pred_prob_cb >= 0.5).astype(int)
    f05_cb = fbeta_score(y_val, y_pred_cb, beta=0.5, zero_division=0)
    
    print("\n" + "="*40)
    print("MATCHING ACCURACY (F0.5 SCORE ON VALIDATION)")
    print("="*40)
    print(f"LightGBM:  {f05_lgb:.4f}")
    print(f"CatBoost:  {f05_cb:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_pipeline(sample_size=1000)
