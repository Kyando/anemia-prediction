import json

import dice_ml
import joblib
import pandas as pd
from dice_ml.utils import helpers 

def preprocess_data_for_prediction(df, preprocessor):
    # Separate features and target
    X = df.drop(columns=["Approved"])
    # X = df
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
    # X = df
    y = df["Approved"].values

    # Preprocess the data
    X, y = preprocess_data_for_prediction(df, preprocessor)

    # Retrieve test data
    test_data = X.loc[test_indexes]
    test_labels = y[test_indexes] if y is not None else None

    # Predict with the trained model
    predictions = loaded_model.predict(test_data)
    probabilities = loaded_model.predict_proba(test_data)

    target_feature = 'num__G1'

    d = dice_ml.Data(dataframe=test_data, continuous_features=list(test_data.columns), outcome_name=target_feature)
    m = dice_ml.Model(model=loaded_model, backend="sklearn")
    exp = dice_ml.Dice(d, m)

    # Select a test instance (e.g., a student predicted to fail)
    query_instance = test_data.iloc[0:1]

    # Generate Counterfactuals
    counterfactuals = exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=target_feature)
    counterfactuals.visualize_as_dataframe()

    print("Finished")