import os

from src.data_preprocessing import preprocess


RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCCESSED_DATA_PATH = "data/processed/Processed-Customer-.csv"
RESULTS_DIR = "results"

def main():
    os.makedirs(RESULTS_DIR,exist_ok=True)
    os.makedirs(os.path.dirname(PROCCESSED_DATA_PATH), exist_ok=True)

    #load + preprocessed
    df = preprocess(RAW_DATA_PATH)




if __name__ == "__main__":
    main()