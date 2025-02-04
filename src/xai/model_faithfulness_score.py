import json

import joblib
import numpy as np
import pandas as pd


# Preprocess data using the loaded scaler and encoder
def preprocess_data_for_prediction(df, preprocessor):
    # Separate features and target
    X = df.drop(columns=["Approved"])
    y = df["Approved"].values if "Approved" in df.columns else None

    # Create a DataFrame with the preprocessed data and feature names
    X_preprocessed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
    return X_preprocessed, y


def normalize_importances(explanations, top_k=10):
    total_importance = sum(abs(score) for _, score in explanations)
    normalized_explanations = [(feature, abs(score) / total_importance) for feature, score in explanations]
    sorted_importances = sorted(normalized_explanations, key=lambda x: abs(x[1]), reverse=True)
    return sorted_importances[:top_k]


def calculate_faithfulness(model, X, importance_data, top_k_features, method="shap"):
    """
    Calculate faithfulness metric for a given XAI method.

    Args:
        model: Trained model for prediction.
        X: Dataset corresponding to the samples in the JSON data (numpy array or pandas DataFrame).
        importance_data: Parsed JSON containing feature importance scores.
        method: Method to evaluate ('shap' or 'lime').

    Returns:
        Average faithfulness score across all samples.
    """
    faithfulness_scores = []

    for sample in importance_data:
        # Get the feature importance for the chosen method
        norm_importance = normalize_importances(sample[method], top_k_features)
        feature_importance = {feat: imp for feat, imp in norm_importance}

        # Original prediction
        sample_index = sample["dataset_id"]
        index = sample["index"]
        X_sample = X.loc[sample_index]
        original_prediction = model.predict([X_sample])[0]

        impacts = []
        importances = []

        for feature, importance in feature_importance.items():
            # Find feature index
            # feature_idx = list(feature_importance.keys()).index(feature)

            # Perturb the feature (set it to zero)
            X_perturbed = X_sample.copy()

            if "cat__" in feature:
                base_value = X_perturbed[feature]
                X_perturbed[feature] = 1 if base_value == 0 else 0
            else:
                X_perturbed[feature] = 0

            # Get new prediction and compute impact
            new_prediction = model.predict([X_perturbed])[0]
            impact = abs(original_prediction - new_prediction)

            impacts.append(impact)
            importances.append(abs(importance))

        # Calculate correlation between impacts and importances
        if len(impacts) > 1 and len(importances) > 1:  # Avoid correlation errors
            correlation = np.corrcoef(impacts, importances)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            faithfulness_scores.append(correlation)

    # Return the average faithfulness score
    return np.mean(faithfulness_scores)


def load_model_and_test_data():
    subject = "mat"
    model_name = "xgboost"

    model_filename = f"models/{subject}_{model_name}.pkl"
    preprocessor_filename = f"models/{subject}_{model_name}_preprocessor.pkl"
    test_filename = f"models/{subject}_{model_name}_test.json"
    dataset_path = f"datasets/student-{subject}.csv"

    loaded_model = joblib.load(model_filename)
    preprocessor = joblib.load(preprocessor_filename)
    test_indexes = json.loads(open(test_filename).read())

    df = pd.read_csv(dataset_path)
    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=["G3", "G2"])
    # df = df.drop(columns=["G3", "G2", "G1"])

    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Preprocess the data
    X, y = preprocess_data_for_prediction(df, preprocessor)

    # Retrieve test data
    test_data = X.loc[test_indexes]
    test_labels = y[test_indexes] if y is not None else None
    return loaded_model, test_data, test_labels, test_indexes


if __name__ == '__main__':
    # Load JSON data
    with open("top_10_features/test_data_features_ranked.json", "r") as f:
        data = json.load(f)

    model, X, labels, test_indexes = load_model_and_test_data()
    top_k = 57
    # Calculate faithfulness for SHAP
    shap_faithfulness = calculate_faithfulness(model, X, data, top_k, method="shap")

    # Calculate faithfulness for LIME
    lime_faithfulness = calculate_faithfulness(model, X, data, top_k, method="lime")

    print(f"Faithfulness - SHAP: {shap_faithfulness}")
    print(f"Faithfulness - LIME: {lime_faithfulness}")
