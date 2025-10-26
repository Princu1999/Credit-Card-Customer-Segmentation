from src.main import run_pipeline

if __name__ == "__main__":
    # Place your dataset at data/Customer_Data.csv before running
    summary = run_pipeline()
    print("\nPipeline summary:\n", summary)