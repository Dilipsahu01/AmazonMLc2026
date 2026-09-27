import pandas as pd
from src.blocking.union_blocker import UnionBlocker
import traceback

def test_union():
    try:
        print("Creating mock data...")
        # Create a tiny mock dataframe that mimics S1 and S2
        df_s1 = pd.DataFrame({
            'entity_id': ['S1-1', 'S1-2'],
            'country': ['us', 'us'],
            'norm_name': ['apple inc', 'microsoft corp'],
            'norm_address': ['cupertino ca', 'redmond wa']
        })
        
        df_s2 = pd.DataFrame({
            'entity_id': ['S2-1', 'S2-2', 'S2-3'],
            'country': ['us', 'us', 'us'],
            'norm_name': ['apple incorporated', 'micro soft', 'random company'],
            'norm_address': ['cupertino california', 'redmond', 'nowhere']
        })
        
        print("Initializing UnionBlocker...")
        blocker = UnionBlocker(top_k_tfidf=2, top_k_embed=2, ngram_range=(3, 3))
        
        print("Running UnionBlocker...")
        candidates = blocker.run(df_s1, df_s2, s2_prefix='S2')
        
        print("\n--- TEST SUCCESS ---")
        print(f"Generated {len(candidates)} candidates!")
        print(candidates)
        
    except Exception as e:
        print("\n--- TEST FAILED ---")
        traceback.print_exc()

if __name__ == "__main__":
    test_union()
