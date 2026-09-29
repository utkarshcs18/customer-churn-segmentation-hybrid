# stage -> 0 (clean, encode, scale, split)
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
    df.drop(columns=["customerID"], inplace=True)

    df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    binary_columns = ['gender','Partner','Dependents',
                      'PhoneService','PaperlessBilling','Churn']

    le = LabelEncoder()
    for col in binary_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])



    multi_value_columns = ['MultipleLines','InternetService','OnlineSecurity',
                           'OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod']

    df = pd.get_dummies(df, columns=multi_value_columns, drop_first=True, dtype=int)
    return df


def split_features(df : pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    return X,y


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)   

    return X_train_scaled, X_test_scaled, scaler


def preprocess(raw_path:str):
    df = load_data(raw_path)
    inspect_data(df)

    df = clean_data(df)
    df = encode_features(df)

    X, y = split_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, scaler