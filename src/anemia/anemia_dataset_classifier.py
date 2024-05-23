import json

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import utils.data_utils as data_utils
import utils.model_utils as model_utils

if __name__ == '__main__':
    csv_file = "resources/anemia_disease.csv"
    df = data_utils.load_dataset(csv_file)
    # df['Anemia'] = df['Anemia'].apply(lambda x: 1 if x > 0 else 0)
    train_X, train_Y, test_X, test_Y, _ = data_utils.split_train_test_data(data_frame=df, test_size=0.3,
                                                                           y_class="ANEMIA")

    # labels = ["No Anemia", "Anemia"]
    labels = ["No\nAnemia", "HGB\nAnemia", "Iron\nAnemia", "Folate\nAnemia", "B12\nAnemia"]
    models = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN", "NaiveBayes"]

    metrics_list = []
    for model in models:
        y_pred, trained_model = model_utils.train_model(train_X, train_Y, test_X, test_Y, model_name=model)
        output_plot_file = f"output/{model}_confusion_matrix.png"
        output_metrics_file = f"output/{model}_metrics.png"
        model_metrics = model_utils.plot_confusion_matrix(test_Y=test_Y, y_pred=y_pred, labels=labels, model_name=model,
                                                          output_file=output_plot_file,
                                                          metrics_file=output_metrics_file)
        metrics_list.append(model_metrics)
        if model == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(trained_model, filled=True, feature_names=df.columns, class_names=labels, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"output/{model}_tree.png")
            plt.close()

    with open("figures/model_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(metrics_list, indent=2))

    print("finished")
