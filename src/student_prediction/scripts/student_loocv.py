import json

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset_path = "../datasets/student-por.csv"
    df = pd.read_csv(dataset_path)
    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=["G3", "G2"])

    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # List of models to evaluate
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(max_iter=200),
    }

    # Cross-validation setup (Leave-One-Out Cross-Validation)
    loo = LeaveOneOut()

    scores_json = {}

    # Evaluate each model using LOO-CV
    for name, model in models.items():
        # For classification, use SMOTE in a pipeline
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        # Perform cross-validation
        scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')
        scores_json[name] = scores.tolist()
        print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    with open("metrics/loocv_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(scores_json, indent=2))
