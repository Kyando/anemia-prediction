import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

import utils.data_utils as data_utils
import utils.model_utils as model_utils

if __name__ == '__main__':
    dataset_path = "datasets/student-por.csv"
    df = pd.read_csv(dataset_path)

    df["Approved"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)
    df, preprocessor = data_utils.preprocess_and_generate_synthetic_data(base_df=df, target_column="Approved")
    df = df.drop(columns=["G3", "G2"])
    train_X, train_Y, test_X, test_Y, _ = data_utils.split_train_test_data(data_frame=df, test_size=0.3,
                                                                           y_class="Approved")

    labels = ["Not Approved", "Approved"]
    models = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN", "NaiveBayes"]

    metrics_list = []
    for model in models:
        y_pred, trained_model = model_utils.train_model(train_X, train_Y, test_X, test_Y, model_name=model)
        output_plot_file = f"output/{model}_confusion_matrix.png"
        output_metrics_file = f"output/{model}_metrics.png"
        model_metrics = model_utils.plot_confusion_matrix(test_Y=test_Y, y_pred=y_pred, labels=labels, model_name=model,
                                                          output_file=output_plot_file,
                                                          metrics_file=output_metrics_file,
                                                          plot_metrics_figure=False,
                                                          percentage=False)
        metrics_list.append(model_metrics)
        if model == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(trained_model, filled=True, feature_names=df.columns, class_names=labels, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"output/{model}_tree.png")
            plt.close()

    with open("metrics/model_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(metrics_list, indent=2))

    print("finished")
