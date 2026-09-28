# stage -> 0 (clean, encode, scale, split)

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    print("Sample Data:")
    print(df.head())

    print("\nDimensions:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nInfo:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop customer ID - not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges is stored as string in this dataset - convert to numeric
    # errors="coerce" turns blank strings into NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # fill NaN with median (only 11 rows affected)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Binary columns: Yes/No -> 1/0 using LabelEncoder ---
    binary_columns = [
        "gender", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling", "Churn"
    ]

    le = LabelEncoder()
    for col in binary_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # --- Multi-value columns: one-hot encode ---
    # drop_first=True avoids dummy variable trap (multicollinearity)
    # dtype=int keeps the output as 0/1 integers instead of booleans
    multi_value_columns = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]

    df = pd.get_dummies(df, columns=multi_value_columns, drop_first=True, dtype=int)

    return df


def split_features(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])  # all feature columns
    y = df["Churn"]                 # target label (0 = No, 1 = Yes)
    return X, y


def scale_features(X_train, X_test):
    # fit scaler on training data only — never on test data (avoids data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    # transform only — no fit
    return X_train_scaled, X_test_scaled, scaler


def preprocess(raw_path: str):
    # Step 1: load
    df = load_data(raw_path)

    # Step 2: inspect
    inspect_data(df)

    # Step 3: clean (fix TotalCharges, drop customerID)
    df = clean_data(df)

    # Step 4: encode (binary + one-hot)
    df = encode_features(df)

    # Step 5: split into features and label
    X, y = split_features(df)

    # Step 6: train/test split (80/20, stratified to preserve churn ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 7: scale using scale_features()
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, scaler