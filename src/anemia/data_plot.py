import json

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import table

from src.anemia.utils import data_utils


def plot_table(summary_df, output_file: str, title=""):
    # Display the summary DataFrame
    print(summary_df)
    # Plotting the table
    fig, ax = plt.subplots(figsize=(9, 2))  # set size frame
    ax.axis('tight')
    ax.axis('off')
    columns_width = [0.24] * len(summary_df.columns)
    # columns_width[1] = 0.47
    tbl = table(ax, summary_df, loc='center', cellLoc='center', colWidths=columns_width)
    # Styling the table
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.2)
    # Bold the headers
    for key, cell in tbl.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
    plt.title(title, fontsize=14)
    plt.savefig(output_file)


def plot_table_dataset_one():
    csv_file_one = "resources/anemia_disease.csv"
    csv_file_two = "resources/anemia_dataset.csv"
    df = data_utils.load_dataset(csv_file_one)
    # Creating a list to hold the summary information
    summary_list = []
    # Looping through each column to calculate the required statistics

    #
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


def generate_dataset_info_json(output_file: str):
    csv_file_one = "resources/anemia_disease.csv"
    csv_file_two = "resources/anemia_dataset.csv"
    df_one, df_two = data_utils.load_cross_dataset_validation(csv_file_one, csv_file_two)
    df_one['ANEMIA'] = df_one['ANEMIA'].apply(lambda x: 1 if x > 0 else 0)
    df_two['GENDER'] = df_two['GENDER'].apply(lambda x: 1 if x == 0 else 0)

    desc_map = {
        "GENDER": "0-Female, 1-Male",
        "HGB": "Hemoglobin",
        "MCH": "Mean Corpuscular Hemoglobin",
        "MCHC": "Mean Corpuscular Hemoglobin Concentration",
        "MCV": "Mean Corpuscular Volume",
        "ANEMIA": "0-Not Anemic, 1-Anemic",
    }

    d_one_min_max = {
        "GENDER": "",
        "HGB": "",
        "MCH": "",
        "MCHC": "",
        "MCV": "",
        "ANEMIA": "",
    }
    for field in d_one_min_max:
        min_value = round(df_one[field].min(), 2)
        max_value = round(df_one[field].max(), 2)
        d_one_min_max[field] = f"{min_value}-{max_value}"

    d_two_min_max = {
        "GENDER": "",
        "HGB": "",
        "MCH": "",
        "MCHC": "",
        "MCV": "",
        "ANEMIA": "",
    }
    for field in d_two_min_max:
        min_value = round(df_two[field].min(), 2)
        max_value = round(df_two[field].max(), 2)
        d_two_min_max[field] = f"{min_value}-{max_value}"

    print(d_one_min_max)
    print(d_two_min_max)
    print()
    output_json = []
    for field in d_two_min_max:
        output_json.append(
            {
                "Feature": field,
                "Description": desc_map[field],
                "Dataset 1 Ranges": d_two_min_max[field],
                "Dataset 2 Ranges": d_one_min_max[field],
            }
        )

    return output_json


if __name__ == '__main__':
    # plot_table_dataset_one()
    print("hello")
    # base_json = generate_dataset_info_json("figures/dataset_one_info.json")
    #
    # metrics_df = pd.DataFrame(base_json)
    # plot_table(metrics_df, "figures/dataset_descriptions.png",
    #            title="Datasets features and value ranges")
    if True:
        metrics_values = json.loads(open("multi_class_output/model_metrics.json").read())
        # metrics_values = json.loads(open("figures/model_metrics.json").read())
        # metrics_values = json.loads(open("figures/model_metrics_two.json").read())
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
        plot_table(metrics_df, "figures/table_models_metrics_classification_multi_class.png",
        # plot_table(metrics_df, "figures/table_models_metrics_classification_d1.png",
                   title="")
        # plot_table(metrics_df, "figures/table_models_metrics_classification_d2.png", title="Cross-Dataset Validation")
