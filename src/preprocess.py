from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Identificar coloumnas numericas y categoricas, y construir un pipeline de procesamiento, realizando una limpieza y adaptación de la data,
# imputación, escalado y one-hot encoding.

def build_preprocessor():
    num_cols = [
        "age","Medu","Fedu","traveltime","studytime","failures",
        "famrel","freetime","goout","Dalc","Walc","health",
        "absences","G1","G2"
    ]
    
    cat_cols = [
        "school","sex","address","famsize","Pstatus","Mjob","Fjob",
        "reason","guardian","schoolsup","famsup","paid","activities",
        "nursery","higher","internet","romantic"
    ]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return preprocessor
