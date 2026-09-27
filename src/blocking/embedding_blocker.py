import pandas as pd
import numpy as np
import time
import os
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

class EmbeddingBlocker:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', top_k=20, batch_size=256):
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers and faiss-cpu must be installed to use EmbeddingBlocker.")
            
        print(f"Loading Embedding Model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        # We normalize embeddings so Inner Product (IP) equals Cosine Similarity
        self.index = None
        self.s2_ids = None

    def _build_text(self, df: pd.DataFrame) -> list:
        """Combines normalized name and address for semantic embedding."""
        names = df['norm_name'].fillna('')
        addrs = df['norm_address'].fillna('')
        return (names + " " + addrs).tolist()

    def fit(self, df: pd.DataFrame):
        """Encodes the target database and builds the FAISS index."""
        print(f"  Encoding {len(df)} target records for FAISS...")
        texts = self._build_text(df)
        
        t0 = time.time()
        # Encode in batches, normalize for cosine similarity
        embeddings = self.model.encode(texts, batch_size=self.batch_size, 
                                       normalize_embeddings=True, 
                                       show_progress_bar=True,
                                       convert_to_numpy=True)
        print(f"  Encoded targets in {time.time()-t0:.1f}s")
        
        # Build FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner product = Cosine sim (since normalized)
        self.index.add(embeddings)
        self.s2_ids = df['entity_id'].values

    def transform_and_search(self, df_query: pd.DataFrame, s2_prefix='S2') -> pd.DataFrame:
        """Encodes queries and searches the FAISS index for top K."""
        print(f"  Encoding {len(df_query)} query records...")
        texts = self._build_text(df_query)
        
        query_embeddings = self.model.encode(texts, batch_size=self.batch_size, 
                                             normalize_embeddings=True, 
                                             show_progress_bar=True,
                                             convert_to_numpy=True)
                                             
        print(f"  Searching FAISS index for Top {self.top_k}...")
        t0 = time.time()
        scores, indices = self.index.search(query_embeddings, self.top_k)
        print(f"  FAISS search completed in {time.time()-t0:.1f}s")
        
        # Flatten results into DataFrame
        s1_ids = df_query['entity_id'].values
        
        # Expand arrays for DataFrame construction
        num_queries = len(s1_ids)
        
        # Create flat arrays
        flat_s1 = np.repeat(s1_ids, self.top_k)
        flat_s2 = self.s2_ids[indices.flatten()]
        flat_scores = scores.flatten()
        
        # Filter out cases where index is -1 (not enough results in index)
        valid = indices.flatten() != -1
        
        candidates_df = pd.DataFrame({
            's1_entity_id': flat_s1[valid],
            f'{s2_prefix}_entity_id': flat_s2[valid],
            'blocking_score_embed': flat_scores[valid]
        })
        
        return candidates_df
