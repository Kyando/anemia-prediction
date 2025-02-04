import json

import joblib

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, confusion_matrix
from utils import ml_utils
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

import shap


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


if __name__ == '__main__':
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

    # Predict with the trained model
    predictions = loaded_model.predict(test_data)
    probabilities = loaded_model.predict_proba(test_data)

    # ---------- SHAP EXPLANATIONS ----------------

    shap_explainer = shap.TreeExplainer(loaded_model)
    shap_values = shap_explainer.shap_values(test_data)

    test_instance_index = test_indexes[0]

    # Visualize SHAP explanations
    # shap.summary_plot(shap_values, test_data, feature_names=test_data.columns)
    shap.summary_plot(shap_values, test_data, feature_names=test_data.columns, show=False)
    plt.savefig(f"plots/test_{test_instance_index}_shap_summary.png")
    plt.clf()

    # local_test_instance_index = 0  # index of test_indexes, so here it is local since shap was used with test_data
    # shap_values_instance = shap_values[local_test_instance_index]
    # shap.plots.bar(shap.Explanation(values=shap_values_instance,
    #                                 base_values=shap_explainer.expected_value,
    #                                 data=test_data.iloc[local_test_instance_index],
    #                                 feature_names=test_data.columns))
    # plt.savefig(f"plots/test_{test_instance_index}_shap_bar_plot.png")
    # plt.clf()

    save_plots = True
    top_10_features_list = []
    for idx, _ in enumerate(test_data):
        feature_item = {"index": idx,
                        "dataset_id": test_indexes[idx],
                        }
        # local_test_instance_index = 0  # Index of the test instance
        shap_values_instance = shap_values[idx]

        # Create SHAP Explanation
        shap_explanation = shap.Explanation(
            values=shap_values_instance,
            base_values=shap_explainer.expected_value,
            data=test_data.iloc[idx],
            feature_names=test_data.columns
        )

        # Generate and save the bar plot
        if save_plots:
            shap.plots.bar(shap_explanation, show=False)
            plt.savefig(f"plots/test_{idx}_shap_bar_plot.png", bbox_inches="tight")
            plt.clf()

        shap_features = {}
        for i, feature in enumerate(shap_explanation.feature_names):
            shap_features[feature] = float(shap_explanation.values[i])

        sorted_shap_features = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_shap_features = sorted_shap_features
        print(sorted_shap_features)
        feature_item['shap'] = sorted_shap_features


        # ---------- LIME EXPLANATIONS ----------------

        from lime.lime_tabular import LimeTabularExplainer

        # Ensure the training data and test instance are dense arrays
        X_dense = X.to_numpy()  # Convert SparseDataFrame to a dense NumPy array
        feature_names = X.columns.tolist()  # Use preprocessed feature names
        class_names = ["Not Approved", "Approved"]  # Class names

        # Initialize LIME Explainer
        explainer = LimeTabularExplainer(
            training_data=X_dense,  # Dense training data
            feature_names=feature_names,  # Names from preprocessed data
            class_names=class_names,  # Define class names
            discretize_continuous=True
        )

        # Explain a single prediction (e.g., first test instance)
        # idx = 0  # Index of the instance to explain
        explanation = explainer.explain_instance(
            X_dense[idx],  # Dense input instance
            loaded_model.predict_proba,  # Predict function
            num_features=5  # Number of features to include in the explanation
        )

        # Print explanation
        print(f"Label: {y[idx]}\nPrediction: {predictions[idx]}\nProbability: {probabilities[idx]}")
        print(explanation.as_list())  # List format explanation
        # Optional: Save or visualize the explanation
        # explanation.show_in_notebook()

        # ------- Feature weights explanation -----------
        # Explain a single prediction
        explanation = explainer.explain_instance(
            X_dense[idx],  # Dense input instance
            loaded_model.predict_proba,  # Predict function
            labels=(0, 1),
            num_features=len(feature_names)  # Use all features for full relevance scores
        )

        for i in [0]:
            # for i in [0, 1]:
            # Explanation map for class i (assumes binary classification)
            class_explanation_map = explanation.as_map()[i]  # Index i for explained class

            # Extract feature importance values (raw weights)
            feature_importances = {
                feature_names[feature_idx]: weight
                for feature_idx, weight in class_explanation_map
            }

            # Sort features by absolute importance (but keep the original weights for plotting)
            sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
            lime_top_importances = sorted_importances[:10]
            sorted_importances = sorted(lime_top_importances, key=lambda x: abs(x[1]), reverse=False)
            features, weights = zip(*sorted_importances)

            feature_item['lime'] = lime_top_importances

            # Plot with positive/negative weights
            if save_plots:
                plt.figure(figsize=(10, 6))
                plt.barh(features, weights, color=["green" if w > 0 else "red" for w in weights])
                plt.xlabel("Feature Contribution (Raw Weight)", fontsize=12)
                plt.ylabel("Features", fontsize=12)
                plt.title(f"LIME Feature Contribution for Instance {idx}", fontsize=14)
                plt.axvline(0, color="black", linewidth=0.8, linestyle="--")  # Vertical line at 0
                plt.tight_layout()

                plt.savefig(f"plots/test_{idx}_lime_bar_plot_class_{i}.png")
                plt.clf()

        # Save Lime and Shap Features

        top_10_features_list.append(feature_item)
        # Save features separated by instance
        # with open(f"top_10_features/{idx}_lime.json", "w") as json_file:
        #     json_file.write(json.dumps(lime_top_10, indent=2))
        # with open(f"top_10_features/{idx}_shap.json", "w") as json_file:
        #     json_file.write(json.dumps(sorted_shap_features, indent=2))

    with open(f"top_10_features/test_data_features_ranked.json", "w") as json_file:
        json_file.write(json.dumps(top_10_features_list, indent=2))