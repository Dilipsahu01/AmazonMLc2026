import pandas as pd
import gc
from src.pipeline import run_end_to_end_pipeline

def main():
    DATA_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"
    OUTPUT_DIR = "submission_local"
    
    # Run the pipeline locally
    # Note: To run on the full dataset, remove `sample_size=1000`
    print("Starting Local Execution...")
    run_end_to_end_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, sample_size=1000)

if __name__ == "__main__":
    main()
