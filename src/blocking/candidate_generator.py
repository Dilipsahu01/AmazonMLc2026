import numpy as np
import scipy.sparse as sp
import pandas as pd
import time

def get_top_k_sparse(s1_matrix: sp.csr_matrix, s2_matrix: sp.csr_matrix, k: int = 50, batch_size: int = 1000) -> tuple:
    """
    Computes cosine similarity between S1 (queries) and S2 (database) sparse matrices.
    Returns the top-K indices and scores for each query in S1.
    
    This avoids dense matrix explosion by using batched sparse dot products.
    Assumes vectors are already L2 normalized (TfidfVectorizer does this by default).
    """
    num_queries = s1_matrix.shape[0]
    num_db = s2_matrix.shape[0]
    
    # We will store the top K indices and scores for each query
    # Shape: (num_queries, k)
    top_indices = np.zeros((num_queries, k), dtype=np.int32)
    top_scores = np.zeros((num_queries, k), dtype=np.float32)
    
    # Transpose S2 once for fast dot product (CSR * CSC = fast, or CSR * CSR.T = CSR * CSC)
    s2_matrix_T = s2_matrix.T.tocsc()
    
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        batch_s1 = s1_matrix[i:end_idx]
        
        # Compute dot product for the batch: shape (batch_size, num_db)
        # Result is a sparse matrix.
        sim_batch = batch_s1.dot(s2_matrix_T)
        
        # We need to extract the top K elements for each row in the batch.
        # Process directly from CSR structure to save massive memory
        for row_idx in range(sim_batch.shape[0]):
            row_start = sim_batch.indptr[row_idx]
            row_end = sim_batch.indptr[row_idx+1]
            row_scores = sim_batch.data[row_start:row_end]
            row_indices = sim_batch.indices[row_start:row_end]
            
            if len(row_scores) > k:
                # Argpartition on the non-zero elements only!
                top_k_local = np.argpartition(row_scores, -k)[-k:]
                sorted_local = np.argsort(-row_scores[top_k_local])
                final_k = top_k_local[sorted_local]
                
                top_indices[i + row_idx, :k] = row_indices[final_k]
                top_scores[i + row_idx, :k] = row_scores[final_k]
            else:
                # Sort normally
                sorted_local = np.argsort(-row_scores)
                limit = len(row_scores)
                top_indices[i + row_idx, :limit] = row_indices[sorted_local]
                top_scores[i + row_idx, :limit] = row_scores[sorted_local]
                
    return top_indices, top_scores

class CandidateGenerator:
    def __init__(self, top_k=50):
        self.top_k = top_k
        
    def generate_candidates(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, 
                            s1_matrix: sp.csr_matrix, s2_matrix: sp.csr_matrix, 
                            s2_prefix: str = 'S2') -> pd.DataFrame:
        """
        Generates pairs of candidates from S1 and S2.
        Returns a DataFrame with [s1_id, s2_id, similarity_score]
        """
        print(f"  Generating candidates for {len(df_s1)} queries against {len(df_s2)} targets...")
        t0 = time.time()
        
        # Get top K indices and scores
        top_indices, top_scores = get_top_k_sparse(s1_matrix, s2_matrix, k=self.top_k)
        
        print(f"  Computed sparse dot products in {time.time()-t0:.1f}s")
        
        # Flatten results into pairs
        s1_ids = df_s1['entity_id'].values
        s2_ids = df_s2['entity_id'].values
        
        # Repeat S1 ids K times
        s1_ids_repeated = np.repeat(s1_ids, self.top_k)
        
        # Map indices to S2 ids
        s2_ids_flattened = s2_ids[top_indices.flatten()]
        
        # Flatten scores
        scores_flattened = top_scores.flatten()
        
        # Filter out zero scores (no overlap in TF-IDF)
        mask = scores_flattened > 0
        
        candidates_df = pd.DataFrame({
            's1_entity_id': s1_ids_repeated[mask],
            f'{s2_prefix}_entity_id': s2_ids_flattened[mask],
            'blocking_score': scores_flattened[mask]
        })
        
        return candidates_df
