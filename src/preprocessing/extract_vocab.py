import pandas as pd
import json
import re
import os
import time
from ai4bharat.transliteration import XlitEngine

# Paths
BASE = "6ab10eb3b23ba_student_resource/student_resource"
TRAIN_DIR = f"{BASE}/dataset/train"
TEST_DIR = f"{BASE}/dataset/test"
OUTPUT_FILE = "src/preprocessing/indic_vocab.json"

# Unicode blocks to ai4bharat language codes
LANG_BLOCKS = {
    'hi': r'[\u0900-\u097F]',  # Devanagari (Hindi/Marathi)
    'bn': r'[\u0980-\u09FF]',  # Bengali
    'pa': r'[\u0A00-\u0A7F]',  # Gurmukhi (Punjabi)
    'gu': r'[\u0A80-\u0AFF]',  # Gujarati
    'or': r'[\u0B00-\u0B7F]',  # Oriya (Odia)
    'ta': r'[\u0B80-\u0BFF]',  # Tamil
    'te': r'[\u0C00-\u0C7F]',  # Telugu
    'kn': r'[\u0C80-\u0CFF]',  # Kannada
    'ml': r'[\u0D00-\u0D7F]'   # Malayalam
}

def detect_language(word: str) -> str:
    """Detects the language of a word based on its Unicode characters."""
    for lang_code, regex in LANG_BLOCKS.items():
        if re.search(regex, word):
            return lang_code
    return None

def extract_vocab():
    print("1. Extracting unique vocabulary from datasets...")
    vocab_by_lang = {lang: set() for lang in LANG_BLOCKS.keys()}
    
    files = [
        f"{TRAIN_DIR}/train_source2.tsv",
        f"{TRAIN_DIR}/train_source3.tsv",
        f"{TEST_DIR}/test_source2.tsv",
        f"{TEST_DIR}/test_source3.tsv"
    ]
    
    # We use chunking to avoid blowing up memory while scanning all 20M rows
    total_words = 0
    t0 = time.time()
    for file in files:
        print(f"  Scanning {file}...")
        for chunk in pd.read_csv(file, sep='\t', usecols=['business_name', 'business_address'], chunksize=500_000, na_filter=False, dtype=str):
            # Combine all text
            text = (chunk['business_name'] + " " + chunk['business_address']).str.cat(sep=" ")
            words = set(text.split())
            total_words += len(words)
            
            # Filter words that have at least one Indic character
            for word in words:
                lang = detect_language(word)
                if lang:
                    # Clean punctuation from the word
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    if clean_word:
                        vocab_by_lang[lang].add(clean_word)
                        
    print(f"  Total unique Indic words found in {time.time()-t0:.1f}s:")
    for lang, words in vocab_by_lang.items():
        if words:
            print(f"    {lang}: {len(words)} words")
            
    # Now run ML inference on the unique vocabulary
    print("\n2. Running AI4Bharat ML Transliteration on vocabulary...")
    final_dict = {}
    
    t1 = time.time()
    for lang, words in vocab_by_lang.items():
        if not words:
            continue
            
        print(f"  Initializing XlitEngine for {lang}...")
        # Note: This will download the model weights the first time
        engine = XlitEngine(lang, beam_width=4, rescore=False)
        
        word_list = list(words)
        print(f"  Transliterating {len(word_list)} words for {lang}...")
        
        # Transliterate in batches
        batch_size = 1024
        for i in range(0, len(word_list), batch_size):
            batch = word_list[i:i+batch_size]
            try:
                # translit_sentence treats a list of strings as a sentence. 
                # Wait, translit_sentence expects a string. We want to batch transliterate words.
                # Actually ai4bharat provides `engine.translit_word(word, topk=1)` which returns a dict.
                # Let's just loop. It's relatively fast for 100k words.
                for w in batch:
                    # engine.translit_word(w) returns something like {'hi': ['word1', 'word2']}
                    # Wait, no, it transliterates from Roman TO Indic by default!
                    # For Indic to Roman (Reverse Transliteration):
                    # We need to set src_lang="hi", tgt_lang="en" ? 
                    # Actually, ai4bharat XlitEngine does English -> Indic by default.
                    # Wait, we need to check how to do Native-to-Roman.
                    pass
            except Exception as e:
                print(f"Error processing batch: {e}")
                
        # Let's fix the logic below...
