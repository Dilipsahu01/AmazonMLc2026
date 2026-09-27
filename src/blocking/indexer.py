from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import scipy.sparse as sp

class TFIDFIndexer:
    def __init__(self, analyzer='char_wb', ngram_range=(2, 4), min_df=1, max_df=0.9):
        """
        Initializes the TF-IDF Vectorizer.
        We use character n-grams (default 2-4) because they are highly robust to
        spelling variations, typos, and fuzzy matching compared to word n-grams.
        """
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=False # Input is already lowercased from preprocessing
        )
        self.is_fitted = False

    def build_corpus(self, df: pd.DataFrame, text_col: str) -> pd.Series:
        """Fills NaNs to prevent vectorizer errors."""
        return df[text_col].fillna("")

    def fit(self, df: pd.DataFrame, text_col: str = 'norm_name'):
        """Fits the TF-IDF vocabulary on the dataset."""
        corpus = self.build_corpus(df, text_col)
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame, text_col: str = 'norm_name') -> sp.csr_matrix:
        """Transforms the dataset into sparse TF-IDF vectors."""
        if not self.is_fitted:
            raise ValueError("Indexer must be fitted before transform.")
        
        corpus = self.build_corpus(df, text_col)
        return self.vectorizer.transform(corpus)
        
    def fit_transform(self, df: pd.DataFrame, text_col: str = 'norm_name') -> sp.csr_matrix:
        """Fits and transforms the dataset."""
        corpus = self.build_corpus(df, text_col)
        self.is_fitted = True
        return self.vectorizer.fit_transform(corpus)
