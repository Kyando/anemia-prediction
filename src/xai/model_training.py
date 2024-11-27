import json

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np

# Preprocessing Function
def gpt_preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocess features
    encoder = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()

    X_cat = encoder.fit_transform(X[cat_features])
    X_num = scaler.fit_transform(X[num_features])

    # Combine transformed features
    X_processed = pd.DataFrame.sparse.from_spmatrix(X_cat).join(pd.DataFrame(X_num, columns=num_features))
    return X_processed, y, encoder, scaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Identify categorical and numerical columns
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create preprocessing pipelines
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features),
            ('num', numerical_transformer, num_features)
        ])

    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Create a DataFrame with the preprocessed data and feature names
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

    return X_preprocessed, y, preprocessor


if __name__ == "__main__":
    subject = "mat"  # Example subject
    model_name = "xgboost"  # Example model name
    model_filename = f"models/{subject}_{model_name}.pkl"

    dataset_path = f"datasets/student-{subject}.csv"
    df = pd.read_csv(dataset_path)
    # df["Approved"]
    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=["G3", "G2"])

    # Preprocess data
    X, y, preprocessor = preprocess_data(df)

    # Add an index column to the original dataset
    df = df.reset_index()  # Preserve the original index as a column named "index"

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    test_idx = X_test.index

    # Convert column names to strings for compatibility
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Apply SMOTE to the training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define and train model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    # Save the model and preprocessors
    joblib.dump(model, model_filename)
    joblib.dump(preprocessor, f"models/{subject}_{model_name}_preprocessor.pkl")
    # joblib.dump(scaler, f"models/{subject}_scaler.pkl")
    print(f"Model saved to {model_filename}")

    with open(f"models/{subject}_{model_name}_test.json", "w") as json_file:
        json_file.write(json.dumps(test_idx.tolist()))
