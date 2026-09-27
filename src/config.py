import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"

# File Paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# Pipeline Parameters
NUM_WORKERS = os.cpu_count() or 4
CHUNK_SIZE = 100_000

# Blocking Parameters
TFIDF_TOP_K = 50
EMBEDDING_TOP_K = 20

# Model Parameters
SINGLETON_THRESHOLD = 0.5  # To be tuned
MATCH_THRESHOLD = 0.5      # To be tuned
