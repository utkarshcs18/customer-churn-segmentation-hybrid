# Customer Churn Prediction + Segmentation (Hybrid)

A subscription business (telecom, streaming, or SaaS) needs to know which customers are about to leave. Treating every customer the same misses the point: a young high-spender churns for different reasons than an older low-usage customer. This project groups customers into behavioral segments first, then predicts churn inside each segment, and compares that hybrid pipeline to a single model trained on everyone.

## Problem

One global churn model assumes the same features matter for every customer. In practice they do not. The hybrid approach has two stages:

1. **Unsupervised.** Cluster customers into behavioral groups without looking at whether they churned.
2. **Supervised.** Train a separate churn classifier on each segment, and also train one global classifier on the full population.

The comparison answers a practical question: do per-segment models beat the single global model on F1 and recall? Both results are reported side by side. A small gap, or no gap, is a valid finding.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM / Kaggle), stored at `data/raw/Telco-Customer-Churn.csv`.

Each row is one customer. Useful columns include:

| Column | Role |
| --- | --- |
| `tenure` | Months as a customer |
| `MonthlyCharges`, `TotalCharges` | Billing |
| `Contract` | Month-to-month, one year, or two year |
| `InternetService` | DSL, fiber optic, or none |
| `PaymentMethod` | How the customer pays |
| Service flags | Phone, multiple lines, online security, backup, device protection, tech support, streaming TV, streaming movies |
| `Churn` | Target: `Yes` or `No` |

`customerID` is an identifier and is not a feature. `Churn` is the supervised target and is excluded from clustering.

## Pipeline

A new customer row goes in, is assigned a segment, and that segment's model returns a churn label and a probability.

### Stage 1 — K-means segmentation

- **Input:** behavioral features only (tenure, monthly charges, service usage, and related attributes). The `Churn` column is withheld so the clusters are not trained on the answer.
- **Output:** a segment label for each customer (for example Segment 0, 1, 2).

### Stage 2 — Per-segment classifiers

- **Input:** customers inside one segment, with `Churn` as the target.
- **Models:** Logistic Regression, Decision Tree, or SVM (or an ensemble such as Random Forest).
- **Output:** for a new customer, assign the segment from Stage 1, then use that segment's model to predict churn (`Yes` / `No`) and a probability.

### Baseline — one global model

The same classifier family is trained once on all customers, with no segmentation. Metrics for the global model and the per-segment models are compared on the same held-out evaluation, primarily F1 and recall.

## What “done” looks like

1. A customer row is assigned a segment.
2. That segment's model predicts churn probability.
3. This README (or the evaluation report it points to) shows per-segment metrics and the global-model baseline next to each other, with a short conclusion on which approach wins and why.

<!-- ## Repository layout

```
data/raw/Telco-Customer-Churn.csv   # raw telecom churn data
src/data_preprocessing.py           # cleaning, features, train/test split
src/train_model.py                  # K-means segments + global and per-segment models
src/evaluate.py                     # F1, recall, and the hybrid vs global comparison
main.py                             # end-to-end run
``` -->

## License

MIT.
