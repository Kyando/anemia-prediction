import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # subject = "por"
    # subject = "mat"
    subject = "all"
    dataset_path = f"datasets/student-{subject}.csv"
    # dataset_path = "datasets/student-por.csv"
    df = pd.read_csv(dataset_path)
    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    # df = df.drop(columns=["G3", "G2", "G1"])
    df = df.drop(columns=["G3", "G2", ])

    # Drop less relevant columns
    df = df.drop(columns=["Pstatus", "schoolsup", "famsup", "paid", "activities", "internet", ])

    X = df.drop(columns=["Approved"])
    y = df["Approved"].values

    # Sample preprocessing code as provided
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the models
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(),
    }

    # Define the parameter grids
    param_grids = {
        'Decision Tree': {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        },
        'Random Forest': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10],
        },
        'KNN': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__metric': ['euclidean', 'manhattan'],
        },
        'Naive Bayes': {
            # GaussianNB has no hyperparameters to tune
        },
        'SVM': {
            'model__C': [0.1, 1, 10, 100],
            'model__kernel': ['linear', 'rbf', 'poly'],
            'model__gamma': ['scale', 'auto'],
        },
        'Logistic Regression': [
            {'model__penalty': ['l1', 'l2'],
             'model__C': [0.1, 1, 10, 100],
             'model__solver': ['liblinear'],  # 'liblinear' supports 'l1' and 'l2'
             'model__max_iter': [200, 500, 1000]},  # Increase max_iter
            {'model__penalty': ['elasticnet'],
             'model__C': [0.1, 1, 10, 100],
             'model__solver': ['saga'],  # 'saga' supports 'elasticnet'
             'model__l1_ratio': [0.5],  # Elastic-net mixing parameter
             'model__max_iter': [200, 500, 1000]},  # Increase max_iter
        ]
    }

    # Placeholder for the best models and their best parameters
    best_models = {}

    # Loop over models and perform GridSearchCV
    for model_name, model in models.items():
        print(f"Running GridSearchCV for {model_name}...")

        # Create a pipeline combining the preprocessor with the model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])

        # Get the parameter grid for the current model
        param_grid = param_grids.get(model_name, {})

        if param_grid:  # Proceed only if there are parameters to tune
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='accuracy',  # or other metrics like 'f1', 'roc_auc', etc.
                cv=10,  # 5-fold cross-validation
                n_jobs=-1,  # Use all available CPUs
            )

            # Fit the model
            grid_search.fit(X, y)

            # Store the best model and its parameters
            best_models[model_name] = grid_search.best_estimator_

            # Print the best parameters
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_}\n")
        else:
            print(f"No parameters to tune for {model_name}. Skipping...\n")
