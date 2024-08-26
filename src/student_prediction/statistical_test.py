import json
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from scipy.stats import lognorm

if __name__ == '__main__':
    path_A = "statistic_data/por_Logistic Regression_acc.json"
    path_B = "statistic_data/imb_por_Random Forest_acc.json"

    json_A = json.loads(open(path_A).read())
    json_B = json.loads(open(path_B).read())

    data_A = json_A['accuracy']
    data_B = json_B['accuracy']

    plt.hist(data_B, density=True, bins=10)  # density=False would make counts
    plt.ylabel('Counts')
    plt.xlabel('Accuracy')

    plt.show()

    print("Data_B Plot")
    plt.clf()

    # perform Shapiro-Wilk test for normality
    output = shapiro(data_B)
    print(output)
