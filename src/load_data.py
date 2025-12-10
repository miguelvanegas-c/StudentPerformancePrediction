import pandas as pd

# cargar los datasets, concatenarlos y preparar la variable objetivo

def load_datasets():
    mat = pd.read_csv("data/student-mat.csv", sep=";")
    por = pd.read_csv("data/student-por.csv", sep=";")
    df = pd.concat([mat, por], ignore_index=True)
    return df

def prepare_target(df):
    df["target"] = (df["G3"] >= 10).astype(int)
    return df
