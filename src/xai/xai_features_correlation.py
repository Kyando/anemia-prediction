import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def normalize_importances(explanations, top_k=10):
    total_importance = sum(abs(score) for _, score in explanations)
    normalized_explanations = [(feature, abs(score) / total_importance) for feature, score in explanations]
    sorted_importances = sorted(normalized_explanations, key=lambda x: abs(x[1]), reverse=True)
    return sorted_importances[:top_k]


def prepare_features(lime_list, shap_list):
    # Convert to dictionaries for easier handling
    lime_dict = {feature: abs(score) for feature, score in lime_list}
    shap_dict = {feature: abs(score) for feature, score in shap_list}

    # Get the union of features
    all_features = set(lime_dict.keys()).union(set(shap_dict.keys()))

    # Align importance scores
    lime_scores = [lime_dict.get(feature, 0) for feature in all_features]
    shap_scores = [shap_dict.get(feature, 0) for feature in all_features]

    return lime_scores, shap_scores, list(all_features)


if __name__ == '__main__':
    features_rank = json.loads(open("top_10_features/test_data_features_ranked.json").read())

    k = 57
    spearman_correlations = []
    pearson_correlations = []

    # Compute rank correlations for each instance
    for idx, item in enumerate(features_rank):
        lime_explanations = normalize_importances(item['lime'], k)
        shap_explanations = normalize_importances(item['shap'], k)

        lime_scores, shap_scores, features = prepare_features(lime_explanations, shap_explanations)
        spearman_corr, _ = spearmanr(lime_scores, shap_scores)
        pearson_corr, _ = pearsonr(lime_scores, shap_scores)
        spearman_correlations.append(spearman_corr)
        pearson_correlations.append(pearson_corr)

    # Convert to a DataFrame for analysis
    spearman_correlation_df = pd.DataFrame(spearman_correlations, columns=['Spearman Correlation'])

    # Summary Statistics
    mean_corr = spearman_correlation_df['Spearman Correlation'].mean()
    median_corr = spearman_correlation_df['Spearman Correlation'].median()
    print("Spearman Correlation")
    print(f"Mean Correlation: {mean_corr}")
    print(f"Median Correlation: {median_corr}")

    # Plot Histogram
    plt.hist(spearman_correlation_df, bins=10, edgecolor='k')
    plt.title('Distribution of Spearman Rank Correlations')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/top_{k}_spearman_correlation.png")
    plt.clf()

    # Convert to a DataFrame for analysis
    pearson_correlation_df = pd.DataFrame(pearson_correlations, columns=['Pearson Correlation'])

    # Summary Statistics
    mean_corr = pearson_correlation_df['Pearson Correlation'].mean()
    median_corr = pearson_correlation_df['Pearson Correlation'].median()
    print("Pearson Correlation")
    print(f"Mean Correlation: {mean_corr}")
    print(f"Median Correlation: {median_corr}")

    # Plot Histogram
    plt.hist(pearson_correlation_df, bins=10, edgecolor='k')
    plt.title('Distribution of Pearson Correlations')
    plt.xlabel('Pearson Correlation')
    plt.ylabel('Frequency')
    plt.savefig(f"plots/top_{k}_pearson_correlation.png")
    plt.clf()
