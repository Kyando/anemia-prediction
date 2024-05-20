import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

import utils.data_utils as data_utils
import utils.model_utils as model_utils

from sklearn import preprocessing

if __name__ == '__main__':
    csv_file = "resources/anemia_disease.csv"
    df = data_utils.load_dataset(csv_file)
    # df['Anemia'] = df['Anemia'].apply(lambda x: 1 if x > 0 else 0)
    train_X, train_Y, test_X, test_Y, _ = data_utils.split_train_test_data(data_frame=df, test_size=0.3,
                                                                           y_class="Anemia")

    # labels = ["No Anemia", "Anemia"]
    labels = ["No_Anemia", "HGB_Anemia_Class", "Iron_anemia_Class", "Folate_anemia_class", "B12_Anemia_class"]
    models = ["RandomForest", "SVM", "LogisticRegression", "DecisionTree", "KNN", "NaiveBayes"]

    for model in models:
        y_pred, trained_model = model_utils.train_model(train_X, train_Y, test_X, test_Y, model_name=model)
        output_plot_file = f"output/{model}_confusion_matrix.png"
        output_metrics_file = f"output/{model}_metrics.png"
        model_utils.plot_confusion_matrix(test_Y=test_Y, y_pred=y_pred, labels=labels, model_name=model,
                                          output_file=output_plot_file, metrics_file=output_metrics_file)
        if model == "DecisionTree":
            plt.figure(figsize=(20, 10))
            plot_tree(trained_model, filled=True, feature_names=df.columns, class_names=labels, rounded=True)
            plt.title("Decision Tree")
            plt.savefig(f"output/{model}_tree.png")
            plt.close()

    print("finished")
