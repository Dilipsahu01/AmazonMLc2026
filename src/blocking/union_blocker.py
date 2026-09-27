import pandas as pd
from src.blocking.pipeline import BlockingPipeline
from src.blocking.embedding_blocker import EmbeddingBlocker

class UnionBlocker:
    def __init__(self, top_k_tfidf=20, top_k_embed=20, ngram_range=(3, 3)):
        print("Initializing Union Blocker (TF-IDF + Semantic Embeddings)")
        self.tfidf_blocker = BlockingPipeline(top_k=top_k_tfidf, ngram_range=ngram_range)
        self.embed_blocker = EmbeddingBlocker(top_k=top_k_embed)
        
    def run(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, s2_prefix='S2') -> pd.DataFrame:
        """Runs both blockers per country partition, unions results, and deduplicates."""
        all_candidates = []
        
        # We must partition by country, just like BlockingPipeline does, 
        # to ensure we don't match a US entity to an India entity.
        countries = [c for c in df_s1['country'].unique() if pd.notna(c)]
        
        for country in countries:
            df_s1_part = df_s1[df_s1['country'] == country]
            df_s2_part = df_s2[df_s2['country'] == country]
            
            if len(df_s1_part) == 0 or len(df_s2_part) == 0:
                continue
                
            print(f"--- Processing partition: {country} (S1: {len(df_s1_part)}, {s2_prefix}: {len(df_s2_part)}) ---")
            
            # 1. TF-IDF Blocking
            print("  [TF-IDF Phase]")
            cand_tfidf = self.tfidf_blocker._block_partition(df_s1_part, df_s2_part, s2_prefix)
            
            # 2. Embedding Blocking
            print("  [Semantic Embedding Phase]")
            self.embed_blocker.fit(df_s2_part)
            cand_embed = self.embed_blocker.transform_and_search(df_s1_part, s2_prefix)
            
            # 3. Union and Deduplicate
            combined = pd.concat([cand_tfidf, cand_embed], ignore_index=True)
            
            # If a pair was found by both, keep the max score?
            # Actually, just drop duplicate pairs.
            # We don't really care about the raw blocking scores in the downstream ML model
            # because the ML model computes exact Jaro-Winkler anyway.
            # But we can keep the first occurrence.
            
            combined_dedup = combined.drop_duplicates(subset=['s1_entity_id', f'{s2_prefix}_entity_id'], keep='first')
            
            print(f"  Union complete. Lexical: {len(cand_tfidf)}, Semantic: {len(cand_embed)}, Merged Unique: {len(combined_dedup)}")
            all_candidates.append(combined_dedup)
            
        if all_candidates:
            return pd.concat(all_candidates, ignore_index=True)
        else:
            return pd.DataFrame(columns=['s1_entity_id', f'{s2_prefix}_entity_id'])
