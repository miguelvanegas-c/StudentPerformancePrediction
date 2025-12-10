from load_data import load_datasets, prepare_target
from preprocess import build_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib

def train():
    df = load_datasets()
    df = prepare_target(df)

    X = df.drop(columns=["G3", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()

    pipe = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    params = {
        "rf__n_estimators": [100,200,300,500],
        "rf__max_depth": [None,10,20,30],
        "rf__min_samples_split": [2,5,10],
        "rf__min_samples_leaf": [1,2,4]
    }

    search = RandomizedSearchCV(
        pipe, params, n_iter=20, scoring="f1", cv=5, n_jobs=-1, random_state=42
    )

    search.fit(X_train, y_train)

    joblib.dump(search.best_estimator_, "models/rf_best.joblib")

    print("Modelo guardado en models/rf_best.joblib")

if __name__ == "__main__":
    train()
