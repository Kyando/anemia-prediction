import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train_model(train_X, train_Y, test_X, test_Y, model_name="RandomForest", **kwargs):
    if model_name == "RandomForest":
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 10)
        min_samples_leaf = kwargs.get('min_samples_leaf', 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf, random_state=42)

    elif model_name == "SVM":
        C = kwargs.get('C', 1.0)
        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', 'scale')
        model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)

    elif model_name == "LogisticRegression":
        C = kwargs.get('C', 1.0)
        solver = kwargs.get('solver', 'lbfgs')
        max_iter = kwargs.get('max_iter', 100)
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)

    elif model_name == "DecisionTree":
        max_depth = kwargs.get('max_depth', 5)
        min_samples_split = kwargs.get('min_samples_split', 2)
        min_samples_leaf = kwargs.get('min_samples_leaf', 5)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=42)

    elif model_name == "KNN":
        n_neighbors = kwargs.get('n_neighbors', 5)
        weights = kwargs.get('weights', 'uniform')
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    elif model_name == "NaiveBayes":
        var_smoothing = kwargs.get('var_smoothing', 1e-9)
        model = GaussianNB(var_smoothing=var_smoothing)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    # Train the model
    model.fit(train_X, train_Y)

    # Make predictions
    y_pred = model.predict(test_X)
    return y_pred, model


def plot_confusion_matrix(test_Y, y_pred, labels=["No Anemia", "Anemia"], model_name="model",
                          output_file="output/confusion_matrix.png",
                          metrics_file="output/metrics.png"):
    # Plot confusion matrix
    conf_matrix = confusion_matrix(test_Y, y_pred)
    # Normalize confusion matrix to get percentages
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)  # Increase font scale for better readability
    sns.heatmap(conf_matrix_percentage, annot=True,
                fmt=".3f",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels, cbar=False, annot_kws={"size": 12})

    # Set plot labels and title
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Actual", fontsize=16)
    plt.title(model_name, fontsize=18)
    plt.savefig(output_file)
    plt.close()

    # Evaluate the model
    report = classification_report(test_Y, y_pred, target_names=labels, output_dict=True)
    print(report)

    # Save the classification report as a text plot
    plt.figure(figsize=(6, 2))
    plt.title(model_name)
    # plt.text(0.01, 0.5, str(model_name), {'fontsize': 12}, fontproperties='monospace')
    plt.text(0.01, 0, str(report), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(metrics_file)
    plt.close()

    accuracy = accuracy_score(test_Y, y_pred)
    precision = precision_score(test_Y, y_pred, average='weighted')
    recall = recall_score(test_Y, y_pred, average='weighted')  # Recall is the same as sensitivity
    f1 = f1_score(test_Y, y_pred, average='weighted')

    # Store the metrics in a list
    model_metrics = {
        "Name": model_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Sensitivity': recall,
        'Precision': precision,
    }
    return model_metrics
