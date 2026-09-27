import pandas as pd
import time
from src.data_loader import load_and_preprocess
from src.blocking.pipeline import BlockingPipeline

def evaluate_blocking(sample_size=20000):
    print("Loading datasets...")
    # Load ground truth
    df_gt = pd.read_csv("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv", sep='\t', na_filter=False)
    
    # We will sample S1 to speed up evaluation
    df_s1 = load_and_preprocess("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv")
    df_s2 = load_and_preprocess("6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv")
    
    print(f"\nEvaluating Blocking on a sample of {sample_size} S1 records...")
    df_s1_sample = df_s1.head(sample_size)
    s1_ids = set(df_s1_sample['entity_id'])
    
    # Filter Ground truth for just these S1 records
    df_gt_sample = df_gt[df_gt['source1_entity_id'].isin(s1_ids)]
    
    # Parse ground truth matches
    valid_s2_true_matches = set()
    for _, row in df_gt_sample.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        if matches_str:
            matches = str(matches_str).split(',')
            for match in matches:
                if match.startswith('S2-'):
                    valid_s2_true_matches.add((s1_id, match))
    
    # Filter df_s2 to save memory
    valid_countries = df_s1_sample['country'].unique()
    df_s2_filtered = df_s2[df_s2['country'].isin(valid_countries)]
    
    pipeline = BlockingPipeline(top_k=50, ngram_range=(3, 3))
    candidates_s2 = pipeline.run(df_s1_sample, df_s2_filtered, s2_prefix='S2')
    
    # Evaluate S2 Recall
    if len(candidates_s2) > 0:
        generated_pairs = set(zip(candidates_s2['s1_entity_id'], candidates_s2['S2_entity_id']))
    else:
        generated_pairs = set()
        
    found = 0
    for match in valid_s2_true_matches:
        if match in generated_pairs:
            found += 1
            
    total_true = len(valid_s2_true_matches)
    recall = found / total_true if total_true > 0 else 0
    fnr = 1.0 - recall
    
    print("\n" + "="*40)
    print("BLOCKING EVALUATION RESULTS (TF-IDF Top-50)")
    print("="*40)
    print(f"Total True S2 Matches in Sample: {total_true}")
    print(f"True Matches Found in Top-50:    {found}")
    print(f"Recall @ 50:                     {recall:.4f} ({(recall*100):.2f}%)")
    print(f"False Negative Rate:             {fnr:.4f} ({(fnr*100):.2f}%)")
    print("="*40)

if __name__ == "__main__":
    evaluate_blocking(sample_size=10000)
