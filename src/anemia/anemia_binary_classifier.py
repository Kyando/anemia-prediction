import json

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import utils.data_utils as data_utils
import utils.model_utils as model_utils

if __name__ == '__main__':
    csv_file_one = "resources/anemia_disease.csv"
    csv_file_two = "resources/anemia_dataset.csv"
    df_one, df_two = data_utils.load_cross_dataset_validation(csv_file_one, csv_file_two)
    df_one['ANEMIA'] = df_one['ANEMIA'].apply(lambda x: 1 if x > 0 else 0)
    df_two['GENDER'] = df_two['GENDER'].apply(lambda x: 1 if x == 0 else 1)  # Invert GENDER attributes
    train_X, train_Y, test_X, test_Y, preprocessor = data_utils.split_train_test_data(data_frame=df_one, test_size=0.3,
                                                                                      y_class="ANEMIA")

    x_two_preprocessed, y_two, _ = data_utils.preprocess_data(df_two, "ANEMIA", preprocessor)

    labels = ["No Anemia", "Anemia"]
    # labels = ["No\nAnemia", "HGB\nAnemia", "Iron\nAnemia", "Folate\nAnemia", "B12\nAnemia"]
    models = ["DecisionTree", "KNN", "LogisticRegression", "NaiveBayes", "RandomForest", "SVM"]

    metrics_list = []
    metrics_list_two = []
    for model in models:
        # Dataset one
        y_pred, trained_model = model_utils.train_model(train_X, train_Y, test_X, test_Y, model_name=model)
        output_plot_file = f"output/{model}_confusion_matrix.png"
        output_metrics_file = f"output/{model}_metrics.png"
        model_metrics = model_utils.plot_confusion_matrix(test_Y=test_Y, y_pred=y_pred, labels=labels, model_name=model,
                                                          output_file=output_plot_file,
                                                          metrics_file=output_metrics_file,
                                                          plot_metrics_figure=False,
                                                          percentage=True)
        metrics_list.append(model_metrics)
        if model == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(trained_model, filled=True, feature_names=df_one.columns, class_names=labels, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"output/{model}_tree.png")
            plt.close()

        # Dataset two

        output_plot_file = f"output/{model}_B_confusion_matrix.png"
        output_metrics_file_two = f"output/{model}_B_metrics.png"
        y_pred_two = trained_model.predict(x_two_preprocessed)
        dataset_two_metrics = model_utils.plot_confusion_matrix(test_Y=y_two, y_pred=y_pred_two, labels=labels,
                                                                model_name=model,
                                                                output_file=output_plot_file,
                                                                metrics_file=output_metrics_file,
                                                                plot_metrics_figure=False,
                                                                percentage=True
                                                                )
        metrics_list_two.append(dataset_two_metrics)

    with open("figures/model_metrics.json", "w") as metrics_json:
        metrics_json.write(json.dumps(metrics_list, indent=2))
    with open("figures/model_metrics_two.json", "w") as metrics_json_two:
        metrics_json_two.write(json.dumps(metrics_list_two, indent=2))

    print("finished")
