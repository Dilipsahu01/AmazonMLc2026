import pandas as pd
from src.preprocessing.text_normalizer import normalize_series
from src.preprocessing.name_normalizer import normalize_name_series
from src.preprocessing.country_normalizer import normalize_country_series
from src.blocking.indexer import TFIDFIndexer
from src.blocking.candidate_generator import CandidateGenerator
from src.blocking.country_partition import filter_by_country, get_country_partitions

def run_test():
    # 1. Create Mock Data (already preprocessed style)
    s1_data = {
        'entity_id': ['s1_1', 's1_2'],
        'norm_name': ['dahlia power reliable scientific limited liability company', 'raam marketing private limited'],
        'country': ['us', 'india']
    }
    s2_data = {
        'entity_id': ['s2_1', 's2_2', 's2_3', 's2_4'],
        'norm_name': ['dahlia power scientific llc', 'raam maarketting private limited', 'random business inc', 'ram marketing pvt ltd'],
        'country': ['us', 'india', 'us', 'india']
    }
    
    df_s1 = pd.DataFrame(s1_data)
    df_s2 = pd.DataFrame(s2_data)
    
    print("Mock S1:")
    print(df_s1)
    
    # 2. Fit TF-IDF on S2
    print("\nFitting TF-IDF...")
    indexer = TFIDFIndexer(ngram_range=(3, 3)) # trigrams
    indexer.fit(df_s2)
    
    # 3. Transform S1 and S2
    s1_matrix = indexer.transform(df_s1)
    s2_matrix = indexer.transform(df_s2)
    
    # 4. Generate Candidates
    print("\nGenerating Candidates...")
    generator = CandidateGenerator(top_k=2)
    candidates = generator.generate_candidates(df_s1, df_s2, s1_matrix, s2_matrix)
    
    print("\nCandidates:")
    print(candidates)

if __name__ == "__main__":
    run_test()
