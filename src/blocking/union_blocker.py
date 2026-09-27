import pandas as pd
from src.blocking.pipeline import BlockingPipeline
from src.blocking.embedding_blocker import EmbeddingBlocker

class UnionBlocker:
    def __init__(self, top_k_tfidf=20, top_k_embed=20, ngram_range=(3, 3)):
        print("Initializing Union Blocker (TF-IDF + Semantic Embeddings)")
        self.tfidf_blocker = BlockingPipeline(top_k=top_k_tfidf, ngram_range=ngram_range)
        self.embed_blocker = EmbeddingBlocker(top_k=top_k_embed)
        
    def run(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, s2_prefix='S2') -> pd.DataFrame:
        """Runs both blockers and unions the results."""
        
        # 1. TF-IDF Phase (handles its own country partitioning)
        print("  [TF-IDF Phase]")
        cand_tfidf = self.tfidf_blocker.run(df_s1, df_s2, s2_prefix)
        
        # 2. Embedding Phase (we must partition manually to respect country boundaries)
        print("  [Semantic Embedding Phase]")
        all_embed_cands = []
        countries = [c for c in df_s1['country'].unique() if pd.notna(c)]
        
        for country in countries:
            df_s1_part = df_s1[df_s1['country'] == country]
            df_s2_part = df_s2[df_s2['country'] == country]
            
            if len(df_s1_part) == 0 or len(df_s2_part) == 0:
                continue
                
            self.embed_blocker.fit(df_s2_part)
            cand_embed = self.embed_blocker.transform_and_search(df_s1_part, s2_prefix)
            all_embed_cands.append(cand_embed)
            
        if all_embed_cands:
            cand_embed_full = pd.concat(all_embed_cands, ignore_index=True)
        else:
            cand_embed_full = pd.DataFrame(columns=['s1_entity_id', f'{s2_prefix}_entity_id', 'blocking_score_embed'])
            
        # 3. Union and Deduplicate
        combined = pd.concat([cand_tfidf, cand_embed_full], ignore_index=True)
        combined_dedup = combined.drop_duplicates(subset=['s1_entity_id', f'{s2_prefix}_entity_id'], keep='first')
        
        print(f"  Union complete. Lexical: {len(cand_tfidf)}, Semantic: {len(cand_embed_full)}, Merged Unique: {len(combined_dedup)}")
        return combined_dedup
