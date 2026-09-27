import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

A = sp.csr_matrix([[1, 2], [3, 4]], dtype=np.float32)
B = sp.csr_matrix([[1, 0], [0, 1]], dtype=np.float32)

C = sp_matmul_topn(A, B.T.tocsr(), top_n=1)
print(C.toarray())
print(C.nonzero())
print(C.data)
