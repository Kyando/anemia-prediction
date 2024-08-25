import json
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, f1_score
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

from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    # Training
    subject = "por"
    dataset_path = f"datasets/student-{subject}.csv"
    df = pd.read_csv(dataset_path)
    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=["G3", "G2", ])
    df = df.drop(columns=["Pstatus", "schoolsup", "famsup", "paid", "activities", "internet", ])

    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Validation
    val_subject = "mat"
    dataset_path = f"datasets/student-{val_subject}.csv"
    val_df = pd.read_csv(dataset_path)
    val_df["Approved"] = val_df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    val_df = val_df.drop(columns=["G3", "G2", ])
    val_df = val_df.drop(columns=["Pstatus", "schoolsup", "famsup", "paid", "activities", "internet", ])

    X_val = df.drop(columns=["Approved"])
    y_val = df["Approved"].values

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # List of models to evaluate
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(max_iter=200),
    }

    cross_validation_list = []
    for i in range(20):
        # kf = KFold(n_splits=10, shuffle=True, random_state=i)
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        cross_validation_list.append(kf)
    # cv = loo

    # Initialize lists to store results

    scores_json = {}

    class_names = ["Not Approved", "Approved"]
    for name, model in models.items():
        metrics = []
        confusion_matrices = []

        for i, cv in enumerate(cross_validation_list):

            # Pipeline with SMOTE and model
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=i)),
                ('model', model)
            ])

            # Perform custom cross-validation
            for train_idx, test_idx in cv.split(X, y):
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train the model
                pipeline.fit(X_train, y_train)

                # Make predictions
                y_pred = pipeline.predict(X_val)

                # Compute classification report
                report = classification_report(y_val, y_pred, output_dict=True)
                metrics.append(report)

                # Compute confusion matrix
                cm = confusion_matrix(y_val, y_pred)
                confusion_matrices.append(cm)

        # Initialize dictionaries to store sums for averaging
        accuracy_sums = {cls: 0.0 for cls in np.unique(y)}
        precision_sums = {cls: 0.0 for cls in np.unique(y)}
        recall_sums = {cls: 0.0 for cls in np.unique(y)}
        f1_sums = {cls: 0.0 for cls in np.unique(y)}
        support_sums = {cls: 0 for cls in np.unique(y)}

        # Count the number of folds
        num_folds = len(metrics)

        # Accumulate the metrics for each class across all folds
        for metric in metrics:
            for cls in np.unique(y):
                accuracy_sums[cls] += metric['accuracy']
                cls_str = str(cls)
                if cls_str in metric:
                    precision_sums[cls] += metric[cls_str]['precision']
                    recall_sums[cls] += metric[cls_str]['recall']
                    f1_sums[cls] += metric[cls_str]['f1-score']
                    support_sums[cls] += metric[cls_str]['support']

        # Calculate the average for each metric
        average_metrics = {
            cls: {
                'accuracy': accuracy_sums[cls] / num_folds,
                'precision': precision_sums[cls] / num_folds,
                'recall': recall_sums[cls] / num_folds,
                'f1-score': f1_sums[cls] / num_folds,
                'support': support_sums[cls] / num_folds
            }
            for cls in np.unique(y)
        }

        # Display the average metrics
        for cls, cls_metrics in average_metrics.items():
            print(f"Class {cls} metrics:")
            print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall: {cls_metrics['recall']:.4f}")
            print(f"  F1-Score: {cls_metrics['f1-score']:.4f}")
            print(f"  Support: {cls_metrics['support']:.4f}")
            scores_json[f"{name}_{cls}"] = cls_metrics

        # Aggregate confusion matrices (sum over all folds)
        total_confusion_matrix = np.sum(confusion_matrices, axis=0)
        print("Total Confusion Matrix:")
        print(total_confusion_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(total_confusion_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(f"output/cdv_smote_{subject}_{val_subject}_{name}_cv_confusion_matrix.png")
        plt.clf()

    # with open(f"metrics/imbalanced_{subject}_G1_10k_metrics.json", "w") as metrics_json:
    with open(f"metrics/cdv_smote_{subject}_{val_subject}_G1_10k_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(scores_json, indent=2))
