import os

from src.data_preprocessing import preprocess
from src.train_model import fit_kmeans, train_global_model, train_segment_models
from src.evaluate import evaluate_global_model, evaluate_segment_models, compare_results


RAW_DATA_PATH      = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/Processed-Customer.csv"
RESULTS_DIR        = "results"
N_CLUSTERS         = 3

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # preprocessing
    print("\n" + "="*50)
    print(" Preprocessing")
    print("="*50)

    X_train_scaled, X_test_scaled, y_train, y_test, X_train, scaler = preprocess(RAW_DATA_PATH)
    print(f"\nTrain size : {X_train_scaled.shape[0]} rows, {X_train_scaled.shape[1]} features")
    print(f"Test size  : {X_test_scaled.shape[0]} rows")
    print(f"Churn rate (train): {y_train.mean():.2%}")

    #unsupervised (kmean)
    print("\n" + "="*50)
    print(" K-Means Segmentation")
    print("="*50)

    kmeans, segment_labels = fit_kmeans(X_train_scaled, n_clusters=N_CLUSTERS)
    print(f"\nCustomers segmented into {N_CLUSTERS} behavioral groups.")

    #baseline
    print("\n" + "="*50)
    print(" Global Model (Baseline)")
    print("="*50)

    global_model = train_global_model(X_train_scaled, y_train)

    #hybrid
    print("\n" + "="*50)
    print(" Per-Segment Models (Hybrid)")
    print("="*50)

    segment_models = train_segment_models(X_train_scaled, y_train, segment_labels)
    print(f"\n{len(segment_models)} segment model(s) trained successfully.")


    #evaluation & comparison
    # evaluate global model
    global_metrics = evaluate_global_model(global_model, X_test_scaled, y_test)

    # evaluate segment models
    segment_metrics = evaluate_segment_models(segment_models, kmeans, X_test_scaled, y_test)

    # side-by-side comparison + save report
    compare_results(global_metrics, segment_metrics, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    main()