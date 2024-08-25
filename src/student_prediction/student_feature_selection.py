import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    dataset_path = f"datasets/student-por.csv"

    df = pd.read_csv(dataset_path)

    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3', 'G2', 'G1'])

    # Define features and target
    X = df.drop('Approved', axis=1)
    y = df['Approved']

    # Example: X is your feature matrix and y is your target variable
    # Assume you have categorical and numerical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Column Transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    model = RandomForestClassifier()

    # Pipeline with preprocessing, SMOTE, and RFECV
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE()),  # Apply SMOTE
        ('rfecv', RFECV(estimator=model, step=1, cv=StratifiedKFold(n_splits=10)))
    ])
    pipeline.fit(X, y)

    # Extract transformed feature names
    # Retrieve the feature names from the preprocessor
    transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Apply the RFECV support mask to the transformed feature names
    selected_features = transformed_feature_names[pipeline.named_steps['rfecv'].support_]
    print("Selected features:", list(selected_features))
