import pandas as pd
from typing import Dict, List
import time

from .country_partition import get_country_partitions
from .indexer import TFIDFIndexer
from .candidate_generator import CandidateGenerator

class BlockingPipeline:
    def __init__(self, top_k=50, ngram_range=(2, 4)):
        self.top_k = top_k
        self.ngram_range = ngram_range
        
    def run(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, s2_prefix: str = 's2') -> pd.DataFrame:
        """
        Runs the full blocking pipeline:
        1. Partitions data by country.
        2. Fits TF-IDF on S2 partition.
        3. Transforms S1 and S2 partitions.
        4. Generates top K candidates per query in S1.
        Returns a single unified DataFrame of all candidates.
        """
        print(f"\n--- Starting Blocking Pipeline vs {s2_prefix.upper()} ---")
        t0 = time.time()
        
        # Partition by country
        s1_partitions = get_country_partitions(df_s1)
        s2_partitions = get_country_partitions(df_s2)
        
        all_candidates = []
        
        # We only care about countries present in S1
        for country in s1_partitions.keys():
            df_s1_c = df_s1[s1_partitions[country]]
            
            # If the country doesn't exist in S2, no candidates can be found
            if country not in s2_partitions:
                print(f"Skipping {country}: No records in {s2_prefix.upper()}.")
                continue
                
            df_s2_c = df_s2[s2_partitions[country]]
            print(f"Processing partition: {country} (S1: {len(df_s1_c)}, {s2_prefix.upper()}: {len(df_s2_c)})")
            
            # TF-IDF
            # Note: We fit TF-IDF only on S2 partition to build the vocabulary of the target database
            indexer = TFIDFIndexer(ngram_range=self.ngram_range, min_df=1) # min_df=1 for small partitions
            indexer.fit(df_s2_c)
            
            s1_matrix = indexer.transform(df_s1_c)
            s2_matrix = indexer.transform(df_s2_c)
            
            # Candidate Generation
            generator = CandidateGenerator(top_k=self.top_k)
            candidates = generator.generate_candidates(
                df_s1_c, df_s2_c, s1_matrix, s2_matrix, s2_prefix=s2_prefix
            )
            
            all_candidates.append(candidates)
            
        if not all_candidates:
            return pd.DataFrame()
            
        final_candidates = pd.concat(all_candidates, ignore_index=True)
        print(f"--- Blocking Pipeline Complete: Generated {len(final_candidates)} candidates in {time.time()-t0:.1f}s ---")
        return final_candidates
