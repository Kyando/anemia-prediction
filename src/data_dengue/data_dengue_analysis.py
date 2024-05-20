import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv("resources/data_dengue/data_dengue_sus_202403221029.csv")
print(df.head())

