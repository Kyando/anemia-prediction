import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_and_generate_synthetic_data(base_df, target_column):
    X, y, preprocessor = preprocess_data(base_df, target_column)
    # Apply SMOTE to generate synthetic samples
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Extract the column names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [name.split('__')[-1] for name in feature_names]

    # Convert the resampled data back to a DataFrame with the correct column names
    df_resampled = pd.DataFrame(X_resampled, columns=cleaned_feature_names)
    df_resampled[target_column] = y_resampled
    return df_resampled, preprocessor


def preprocess_data(df, target_column, preprocessor=None):
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    if preprocessor is None:
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Define preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            # ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            # ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y, preprocessor


def split_train_test_data(data_frame, y_class='ANEMIA', test_size=0.3):
    # Preprocess the data
    X_preprocessed, y, preprocessor = preprocess_data(data_frame, target_column=y_class)

    # Split the data into training and testing sets
    train_X, test_X, train_Y, test_Y = train_test_split(X_preprocessed, y, test_size=test_size, random_state=0,
                                                        stratify=y)

    return train_X, train_Y, test_X, test_Y, preprocessor
