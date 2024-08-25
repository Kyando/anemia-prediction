import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_multi_class_anemia_dataset(csv_file: str):
    df = pd.read_csv(csv_file)
    columns = df.columns.values
    print(columns)
    gender = df['GENDER']
    rbc = df['RBC']
    hgb = df['HGB']
    hct = df['HCT']
    mcv = df['MCV']
    mch = df['MCH']
    mchc = df['MCHC']
    rdw = df['RDW']

    plt = df['PLT']
    mpv = df['MPV']
    pdw = df['PDW']
    sd = df['SD']
    sdtsd = df['SDTSD']
    tsd = df['TSD']
    ferretine = df['FERRITTE']
    folate = df['FOLATE']
    b12 = df['B12']

    anemia = df['All_Class']

    data = {
        "GENDER": gender.tolist(),
        "RBC": rbc.tolist(),
        "HGB": hgb.tolist(),
        "HCT": hct.tolist(),
        "MCV": mcv.tolist(),
        "MCH": mch.tolist(),
        "MCHC": mchc.tolist(),
        "RDW": rdw.tolist(),
        "PLT": plt.tolist(),
        "MPV": mpv.tolist(),
        "PDW": pdw.tolist(),
        "SD": sd.tolist(),
        "SDTSD": sdtsd.tolist(),
        "TSD": tsd.tolist(),
        "FERRITTE": ferretine.tolist(),
        "FOLATE": folate.tolist(),
        "B12": b12.tolist(),
        "ANEMIA": anemia.tolist(),
    }

    df = pd.DataFrame(data)
    return df


def load_dataset(csv_file: str):
    df = pd.read_csv(csv_file)
    columns = df.columns.values
    print(columns)
    gender = df['GENDER']
    rbc = df['RBC']
    hgb = df['HGB']
    hct = df['HCT']
    mcv = df['MCV']
    mch = df['MCH']
    mchc = df['MCHC']
    rdw = df['RDW']
    plt = df['PLT']
    mpv = df['MPV']
    pdw = df['PDW']
    sd = df['SD']
    sdtsd = df['SDTSD']
    tsd = df['TSD']
    ferretine = df['FERRITTE']
    folate = df['FOLATE']
    b12 = df['B12']

    anemia = df['All_Class']

    data = {
        "GENDER": gender.tolist(),
        "RBC": rbc.tolist(),
        "HGB": hgb.tolist(),
        "HCT": hct.tolist(),
        "MCV": mcv.tolist(),
        "MCH": mch.tolist(),
        "MCHC": mchc.tolist(),
        "RDW": rdw.tolist(),
        "PLT": plt.tolist(),
        "MPV": mpv.tolist(),
        "PDW": pdw.tolist(),
        "SD": sd.tolist(),
        "SDTSD": sdtsd.tolist(),
        "TSD": tsd.tolist(),
        "FERRITTE": ferretine.tolist(),
        "FOLATE": folate.tolist(),
        "B12": b12.tolist(),
        "ANEMIA": anemia.tolist(),
    }

    df = pd.DataFrame(data)
    return df


def load_cross_dataset_validation(dataset_csv_one: str, dataset_csv_two: str):
    df_one = pd.read_csv(dataset_csv_one)
    columns = df_one.columns.values
    print(columns)
    gender = df_one['GENDER']
    hgb = df_one['HGB']
    mch = df_one['MCH']
    mchc = df_one['MCHC']
    mcv = df_one['MCV']
    anemia = df_one['All_Class']

    data = {
        "GENDER": gender.tolist(),
        "HGB": hgb.tolist(),
        "MCV": mcv.tolist(),
        "MCH": mch.tolist(),
        "MCHC": mchc.tolist(),
        "ANEMIA": anemia.tolist(),
    }

    df_one = pd.DataFrame(data)

    # DF_two
    df_two = pd.read_csv(dataset_csv_two)
    gender = df_two['GENDER']
    hgb = df_two['Hemoglobin']
    mch = df_two['MCH']
    mchc = df_two['MCHC']
    mcv = df_two['MCV']
    anemia = df_two['Result']
    data = {
        "GENDER": gender.tolist(),
        "HGB": hgb.tolist(),
        "MCV": mcv.tolist(),
        "MCH": mch.tolist(),
        "MCHC": mchc.tolist(),
        "ANEMIA": anemia.tolist(),
    }
    df_two = pd.DataFrame(data)

    return df_one, df_two


def preprocess_data(df, target_column, preprocessor=None):
    # Convert numerical categorical columns to category dtype
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    if preprocessor is None:
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Define preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y, preprocessor


def split_train_test_data(data_frame, y_class='ANEMIA', test_size=0.3):
    # Preprocess the data
    X_preprocessed, y, preprocessor = preprocess_data(data_frame, target_column=y_class)

    # Split the data into training and testing sets
    train_X, test_X, train_Y, test_Y = train_test_split(X_preprocessed, y, test_size=test_size, random_state=0,
                                                        stratify=y)

    return train_X, train_Y, test_X, test_Y, preprocessor


def _split_train_test_data(data_frame, y_class='ANEMIA', test_size=0.3):
    data = data_frame
    train, test = train_test_split(data, test_size=test_size, random_state=0, stratify=data[y_class])
    train_X = train[train.columns[:-1]].values
    train_Y = train[train.columns[-1:]].values
    test_X = test[test.columns[:-1]].values
    test_Y = test[test.columns[-1:]].values
    X = data[data.columns[:-1]]
    Y = data['ANEMIA']
    len(train_X), len(train_Y), len(test_X), len(test_Y)
    return train_X, train_Y, test_X, test_Y
