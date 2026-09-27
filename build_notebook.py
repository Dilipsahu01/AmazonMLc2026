import json
import glob
import os

files = [
    'src/config.py',
    'src/preprocessing/__init__.py',
    'src/preprocessing/transliteration.py',
    'src/preprocessing/text_normalizer.py',
    'src/preprocessing/country_normalizer.py',
    'src/preprocessing/name_normalizer.py',
    'src/preprocessing/address_parser.py',
    'src/data_loader.py',
    'src/blocking/country_partition.py',
    'src/blocking/indexer.py',
    'src/blocking/candidate_generator.py',
    'src/blocking/pipeline.py',
    'src/blocking/embedding_blocker.py',
    'src/blocking/union_blocker.py',
    'src/features/name_features.py',
    'src/features/address_features.py',
    'src/features/phonetic_features.py',
    'src/features/cross_features.py',
    'src/features/feature_pipeline.py',
    'src/matching/lgbm_matcher.py',
    'src/matching/catboost_matcher.py',
    'src/matching/singleton_detector.py',
    'src/postprocessing/submission_generator.py',
    'src/pipeline.py'
]

cells = []

# Cell 1: Instructions
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Amazon ML Challenge: Entity Resolution\n",
        "This notebook contains the complete End-to-End Pipeline.\n",
        "It reconstructs the local `src/` modular architecture directly on the Kaggle environment."
    ]
})

# Cell 2: Create directories
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!mkdir -p src/preprocessing src/blocking src/evaluation src/features src/matching src/postprocessing\n",
        "!touch src/__init__.py src/blocking/__init__.py src/evaluation/__init__.py src/features/__init__.py src/matching/__init__.py src/postprocessing/__init__.py\n",
        "!pip install unidecode scikit-learn pandas scipy thefuzz jellyfish lightgbm catboost sparse-dot-topn sentence-transformers faiss-cpu"
    ]
})

# Code cells for each file
for filepath in files:
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r') as f:
        content = f.read()
    
    source_lines = [f"%%writefile {filepath}\n"]
    source_lines.extend([line + "\n" for line in content.split("\n")])
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

# Final Execution Cell
execution_code = """
# KAGGLE EXECUTION SCRIPT
import pandas as pd
from src.pipeline import run_end_to_end_pipeline

# NOTE: Change this to your Kaggle Dataset path!
# Example: DATA_DIR = "/kaggle/input/amazon-ml-challenge-2026/dataset"
DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
OUTPUT_DIR = "submission"

# We run on a 10,000 record sample for testing. 
# Remove sample_size=10000 for the full run!
final_matches = run_end_to_end_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, sample_size=10000)

print(final_matches.head())
"""
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in execution_code.strip().split("\n")]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_er_pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Generated kaggle_er_pipeline.ipynb")
