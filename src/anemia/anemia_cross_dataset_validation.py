import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

import utils.data_utils as data_utils
import utils.model_utils as model_utils

from sklearn import preprocessing

if __name__ == '__main__':
    csv_file_one = "resources/anemia_disease.csv"
    csv_file_two = "resources/anemia_dataset.csv"
    df_one, df_two = data_utils.load_cross_dataset_validation(csv_file_one, csv_file_two)
    df_one['Anemia'] = df_one['Anemia'].apply(lambda x: 1 if x > 0 else 0)
    train_X, train_Y, test_X, test_Y, preprocessor = data_utils.split_train_test_data(data_frame=df_one, test_size=0.3,
                                                                                      y_class="Anemia")

    x_two_preprocessed, y_two, _ = data_utils.preprocess_data(df_two, "Anemia", preprocessor)

    labels = ["No Anemia", "Anemia"]
    # labels = ["No_Anemia", "HGB_Anemia_Class", "Iron_anemia_Class", "Folate_anemia_class", "B12_Anemia_class"]
    # models = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN", "NaiveBayes"]
    models = ["DecisionTree"]
    model = "DecisionTree"
    model_name = "Dataset_A_DecisionTree_A"

    y_pred, trained_model = model_utils.train_model(train_X, train_Y, test_X, test_Y, model_name=model)
    output_plot_file = f"output/{model_name}_confusion_matrix.png"
    output_metrics_file = f"output/{model_name}_metrics.png"
    model_utils.plot_confusion_matrix(test_Y=test_Y, y_pred=y_pred, labels=labels, model_name=model,
                                      output_file=output_plot_file, metrics_file=output_metrics_file)

    plt.figure(figsize=(20, 10))
    plot_tree(trained_model, filled=True, feature_names=df_one.columns, class_names=labels, rounded=True)
    plt.title("Decision Tree")
    plt.savefig(f"output/{model}_tree.png")
    plt.close()

    model_name = "Dataset_B_DecisionTree_A"
    output_plot_file = f"output/{model_name}_confusion_matrix.png"
    output_metrics_file = f"output/{model_name}_metrics.png"

    y_pred_two = trained_model.predict(x_two_preprocessed)
    model_utils.plot_confusion_matrix(test_Y=y_two, y_pred=y_pred_two, labels=labels, model_name=model,
                                      output_file=output_plot_file, metrics_file=output_metrics_file)

    plt.figure(figsize=(20, 10))
    plot_tree(trained_model, filled=True, feature_names=df_two.columns, class_names=labels, rounded=True)
    plt.title("Decision Tree")
    plt.savefig(f"output/{model_name}_tree.png")
    plt.close()

    print("finished model A")

    train_X_B, train_Y_B, test_X_B, test_Y_B, preprocessor_B = data_utils.split_train_test_data(data_frame=df_two,
                                                                                                test_size=0.3,
                                                                                                y_class="Anemia")
    x_one_preprocessed, y_one, _ = data_utils.preprocess_data(df_one, "Anemia", preprocessor_B)

    model = "DecisionTree"
    model_name = "Dataset_B_DecisionTree_B"
    y_pred_B, trained_model_B = model_utils.train_model(train_X_B, train_Y_B, test_X_B, test_Y_B, model_name=model)

    output_plot_file = f"output/{model_name}_confusion_matrix.png"
    output_metrics_file = f"output/{model_name}_metrics.png"
    model_utils.plot_confusion_matrix(test_Y=test_Y_B, y_pred=y_pred_B, labels=labels, model_name=model,
                                      output_file=output_plot_file, metrics_file=output_metrics_file)

    plt.figure(figsize=(20, 10))
    plot_tree(trained_model_B, filled=True, feature_names=df_two.columns, class_names=labels, rounded=True)
    plt.title("Decision Tree")
    plt.savefig(f"output/{model}_tree.png")
    plt.close()

    model_name = "Dataset_A_DecisionTree_B"
    output_plot_file = f"output/{model_name}_confusion_matrix.png"
    output_metrics_file = f"output/{model_name}_metrics.png"

    y_pred_one = trained_model.predict(x_one_preprocessed)
    model_utils.plot_confusion_matrix(test_Y=y_one, y_pred=y_pred_one, labels=labels, model_name=model,
                                      output_file=output_plot_file, metrics_file=output_metrics_file)

    plt.figure(figsize=(20, 10))
    plot_tree(trained_model, filled=True, feature_names=df_one.columns, class_names=labels, rounded=True)
    plt.title("Decision Tree")
    plt.savefig(f"output/{model_name}_tree.png")
    plt.close()
