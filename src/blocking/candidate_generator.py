import numpy as np
import scipy.sparse as sp
import pandas as pd
import time
from sparse_dot_topn import sp_matmul_topn

class CandidateGenerator:
    def __init__(self, top_k=50):
        self.top_k = top_k
        
    def generate_candidates(self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, 
                            s1_matrix: sp.csr_matrix, s2_matrix: sp.csr_matrix, 
                            s2_prefix: str = 'S2') -> pd.DataFrame:
        """
        Generates pairs of candidates from S1 and S2 using sparse_dot_topn.
        This avoids dense matrix explosion by computing only the top K similarities in C++.
        """
        print(f"  Generating candidates for {len(df_s1)} queries against {len(df_s2)} targets...")
        t0 = time.time()
        
        # Ensure matrices are float32 to save memory
        s1_matrix = s1_matrix.astype(np.float32)
        s2_matrix = s2_matrix.astype(np.float32)
        
        # We need S2 transposed. sp_matmul_topn works well with B as CSR.
        s2_matrix_T = s2_matrix.T.tocsr()
        
        # Compute Top K
        # threshold=0.0 means we keep anything > 0.0 (no overlap = discarded)
        print(f"  Running sparse_dot_topn for top {self.top_k}...")
        C = sp_matmul_topn(s1_matrix, s2_matrix_T, top_n=self.top_k, threshold=0.0001, n_threads=4)
        
        print(f"  Computed sparse dot products in {time.time()-t0:.1f}s")
        
        # C is a CSR matrix of shape (num_queries, num_db)
        # We extract the non-zero indices. 
        # C.nonzero() returns (row_indices, col_indices)
        row_indices, col_indices = C.nonzero()
        scores = C.data
        
        # Map indices back to entity IDs
        s1_ids = df_s1['entity_id'].values
        s2_ids = df_s2['entity_id'].values
        
        candidates_df = pd.DataFrame({
            's1_entity_id': s1_ids[row_indices],
            f'{s2_prefix}_entity_id': s2_ids[col_indices],
            'blocking_score': scores
        })
        
        return candidates_df
