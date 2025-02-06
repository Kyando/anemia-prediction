import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

import pandas as pd
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils



# Load dataset
dataset_path = "datasets/student-mat.csv"
df = pd.read_csv(dataset_path, delimiter=",")

# Convert categorical variables to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Store column names
feature_names = df_encoded.columns.tolist()  # Save feature names for later

# Ensure data is numeric
data = df_encoded.apply(pd.to_numeric, errors='coerce').dropna().to_numpy().astype(np.float64)

# Run causal discovery with feature names
cg = pc(data=data, independence_test_method=fisherz, alpha=0.05)
cg.draw_pydot_graph(labels=feature_names)  # Pass feature names as labels

print("Finished")