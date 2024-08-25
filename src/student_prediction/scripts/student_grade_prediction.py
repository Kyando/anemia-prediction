import pandas as pd
import numpy as np

if __name__ == '__main__':
    dataset_path = "../datasets/xAPI-Edu-Data.csv"

    df = pd.read_csv(dataset_path)
    df.head(4)

