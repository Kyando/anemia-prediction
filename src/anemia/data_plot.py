import json

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import table

from src.anemia.utils import data_utils


def plot_table(summary_df, output_file: str):
    # Display the summary DataFrame
    print(summary_df)
    # Plotting the table
    fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
    ax.axis('tight')
    ax.axis('off')
    tbl = table(ax, summary_df, loc='center', cellLoc='center', colWidths=[0.24] * len(summary_df.columns))
    # Styling the table
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.2)
    # Bold the headers
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
    plt.title('', fontsize=14)
    plt.savefig(output_file)


def plot_table_dataset_one():
    csv_file_one = "resources/anemia_disease.csv"
    csv_file_two = "resources/anemia_dataset.csv"
    df = data_utils.load_dataset(csv_file_one)
    # Creating a list to hold the summary information
    summary_list = []
    # Looping through each column to calculate the required statistics
    for column in df.columns:
        summary_list.append({
            'name': column,
            'min': round(df[column].min(), 2),
            'max': df[column].max(),
            'mean': round(df[column].mean(), 2)
        })
    # Converting the list of dictionaries to a DataFrame
    summary_df = pd.DataFrame(summary_list)

    plot_table(summary_df, "figures/table_dataset_one_features.png")


if __name__ == '__main__':
    # plot_table_dataset_one()
    print("hello")
    metrics_values = json.loads(open("figures/model_metrics.json").read())
    model_values = []
    for value in metrics_values:
        model_values.append({
            "model": value['Name'],
            "accuracy": round(value['Accuracy'], 4),
            "f1-score": round(value['F1-Score'], 4),
            "sensitivity": round(value['Sensitivity'], 4),
            "precision": round(value['Precision'], 4),
        })

    metrics_df = pd.DataFrame(model_values)
    plot_table(metrics_df, "figures/table_models_metrics_classification_d1.png")
