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
    df_s2 = load_and_preprocess(f"{data_dir}/train_source2.tsv")
    df_s3 = load_and_preprocess(f"{data_dir}/train_source3.tsv")
    df_gt = pd.read_csv(f"{data_dir}/train_ground_truth.tsv", sep='\t', na_filter=False)
    
    print("\n[1] Blocking (Union TF-IDF + FAISS)...")
    blocker = UnionBlocker(top_k_tfidf=20, top_k_embed=20)
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
    
    # Evaluate Blocking Recall
    found = sum(y)
    total_true = len(true_pairs)
    blocking_recall = found / total_true if total_true > 0 else 0
    print(f"-> Blocking Recall @ Top 20: {blocking_recall:.4f} (Ceiling limit for F0.5)")
    
    print("\n[3] Extracting Features...")
    df_s2_s3 = pd.concat([df_s2, df_s3], ignore_index=True)
    X = build_features(candidates, df_s1, df_s2_s3, s2_prefix='S2')
    
    print("\n[4] Training & Validating LightGBM Model...")
    # Split into 80% train / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    matcher = LGBMMatcher()
    matcher.fit(X_train, y_train)
    
    y_pred_prob = matcher.predict_proba(X_val)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate F0.5
    # precision heavily weighted
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)
    
    print("\n" + "="*40)
    print("MODEL PERFORMANCE (VALIDATION SPLIT)")
    print("="*40)
    print(f"Precision: {precision:.4f} (How many of our predicted matches are correct?)")
    print(f"Recall:    {recall:.4f} (How many true matches did we find?)")
    print(f"F0.5:      {f05:.4f} (Final Leaderboard Metric)")
    print("="*40)

if __name__ == "__main__":
    evaluate_pipeline(sample_size=1000)
