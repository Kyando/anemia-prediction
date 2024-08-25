
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


if __name__ == '__main__':


    # subject = "mat"
    subject = "por"
    # subject = "all"
    dataset_path = f"datasets/student-{subject}.csv"

    df = pd.read_csv(dataset_path)

    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    data = df.drop(columns=['G3', 'G2'])

    # Define features and target
    X = data.drop('Approved', axis=1)
    y = data['Approved']

    # Step 2: Preprocess the data
    # Define which columns are categorical and which are numerical
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Step 3: Handle the imbalance using SMOTE
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Step 4: Train a model with all features using class weights
    pipeline_all_features = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced'))
    ])

    # Train and evaluate the model
    pipeline_all_features.fit(X_train_resampled, y_train_resampled)
    y_pred_all = pipeline_all_features.predict(X_test)
    print("Model with All Features (Class Weight):\n", classification_report(y_test, y_pred_all))

    # Step 5: Apply Feature Selection

    # Option 1: RFE
    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=5, step=1)
    pipeline_rfe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', rfe_selector),
        ('classifier', RandomForestClassifier(class_weight='balanced'))
    ])

    # Train and evaluate the model with RFE-selected features
    pipeline_rfe.fit(X_train_resampled, y_train_resampled)
    y_pred_rfe = pipeline_rfe.predict(X_test)
    print("Model with RFE Selected Features (Class Weight):\n", classification_report(y_test, y_pred_rfe))

    # Option 2: SelectKBest
    kbest_selector = SelectKBest(score_func=f_classif, k=5)
    pipeline_kbest = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', kbest_selector),
        ('classifier', RandomForestClassifier(class_weight='balanced'))
    ])

    # Train and evaluate the model with KBest-selected features
    pipeline_kbest.fit(X_train_resampled, y_train_resampled)
    y_pred_kbest = pipeline_kbest.predict(X_test)
    print("Model with SelectKBest Features (Class Weight):\n", classification_report(y_test, y_pred_kbest))
