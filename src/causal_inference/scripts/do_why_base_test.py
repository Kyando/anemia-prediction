import pandas as pd
from dowhy import CausalModel

import pygraphviz as pgv
print(pgv.__version__)

# Load dataset
dataset_path = "datasets/student-mat.csv"
df = pd.read_csv(dataset_path, delimiter=",")

# Convert categorical variables to numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Store column names
feature_names = df_encoded.columns.tolist()  # Save feature names for later

# Ensure data is numeric
# target_features = ["G3"]
# treatment = ["studytime"]
# common_causes = []
# for i, feature in enumerate(feature_names):
#     if feature in target_features or feature in treatment:
#         continue
#     common_causes.append("x" + str(i))
# data = df_encoded.apply(pd.to_numeric, errors='coerce').dropna().to_numpy().astype(np.float64)
# Define a manually structured causal grap


# Define the causal graph
causal_graph = """
digraph {
    # Student Background
    sex -> G3;
    age -> G3;
    address -> studytime;
    famsize -> studytime;
    Pstatus -> famrel;

    # Parental Influence
    Medu -> studytime;
    Medu -> G3;
    Fedu -> studytime;
    Fedu -> G3;
    Mjob -> studytime;
    Fjob -> studytime;

    # Academic Factors
    school -> G3;
    reason -> studytime;
    guardian -> studytime;
    traveltime -> studytime;
    failures -> G3;

    # Study & Support
    studytime -> G3;
    schoolsup -> G3;
    famsup -> G3;
    paid -> G3;
    activities -> G3;

    # Past Performance
    G1 -> G2;
    G2 -> G3;

    # Social & Lifestyle Factors
    # romantic -> G3;
    # goout -> G3;
    # freetime -> G3;
    # Dalc -> G3;
    # Walc -> G3;
    # health -> G3;
    # absences -> G3;
}
"""

# Create DoWhy model
model = CausalModel(
    data=df,
    graph=causal_graph,
    treatment="studytime",
    outcome="G3"
)

# View the causal graph
model.view_model()
