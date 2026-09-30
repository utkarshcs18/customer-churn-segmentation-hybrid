import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression


def find_optimal_k(x_train_scaled,k_range=range(2,11)):
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_train_scaled)

        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(x_train_scaled, labels))

    return inertias, silhouettes


def fit_kmeans(x_train_scaled, n_clusters: int, random_state: int = 42):

    model = KMeans(n_clusters=n_clusters,random_state=random_state, n_init=10)
    labels = model.fit_predict(x_train_scaled)

    unique, counts = np.unique(labels, return_counts=True)
    for seg, count in zip(unique, counts):
        print(f"Segment {seg}: {count} customers")

    return model, labels


def train_global_model(x_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)

    print(f"\nGlobal Model Trained on {len(y_train)} samples.")

    return model


def train_segment_models(x_train_scaled, y_train, segment_labels):
    segment_models = {}

    y_train_array = np.array(y_train)
    unique_segments = np.unique(segment_labels)

    for seg_id in unique_segments:
        mask = (segment_labels == seg_id)

        x_seg = x_train_scaled[mask]
        y_seg = y_train_array[mask]

        print(f"\nSegment {seg_id}: {len(y_seg)} customers | "
              f"Churn rate: {y_seg.mean():.2%}")


        if len(y_seg)  < 20 or len(np.unique(y_seg)) < 2:
            print(f"Segment {seg_id} skipped — not enough data or only one class.")
            continue

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_seg, y_seg)

        segment_models[seg_id] = model
        print(f" Model trained for Segment {seg_id}")

    return segment_models



def predict_new_customer(new_row_df, kmeans, segment_models, scaler):
    new_scaled = scaler.transform(new_row_df)
    segment = kmeans.predict(new_scaled)[0]

    if segment not in segment_models:
        print(f" No model available for Segment {segment} — was skipped during training.")
        return segment, None, None

    model = segment_models[segment]
    churn_pred = model.predict(new_scaled)[0]
    churn_prob = model.predict_proba(new_scaled)[0][1]

    print(f"Segment: {segment} \nChurn Prediction: {churn_pred} \nChurn Probability: {churn_prob:.2%}")

    return segment, churn_pred, churn_prob
