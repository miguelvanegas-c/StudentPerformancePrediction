import joblib
from load_data import load_datasets, prepare_target
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate():
    model = joblib.load("models/rf_best.joblib")

    df = load_datasets()
    df = prepare_target(df)

    X = df.drop(columns=["G3", "target"])
    y = df["target"]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)

    with open("reports/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"ROC_AUC: {auc}\n")

    print("Reporte generado en reports/metrics.txt")

if __name__ == "__main__":
    evaluate()
