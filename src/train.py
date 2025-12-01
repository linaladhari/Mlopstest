import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import argparse


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y


def build_model(model_type="logistic", k=3):
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=k)
    elif model_type == "logistic":
        return LogisticRegression(max_iter=300)
    else:
        raise ValueError("model_type must be 'knn' or 'logistic'")


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return accuracy, report


def save_metrics(accuracy, report, json_path, txt_path):
    metrics_json = {
        "test_accuracy": accuracy,
        "classification_report": report
    }
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=4)

    with open(txt_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)


def main(data_path, model_path, model_type, k):
    print("Loading data...")
    X, y = load_data(data_path)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training model: {model_type} (k={k})...")
    model = build_model(model_type, k)
    model.fit(X_train, y_train)

    print("Saving model...")
    joblib.dump(model, model_path)

    print("Evaluating...")
    accuracy, report = evaluate_model(model, X_test, y_test)

    print("Saving metrics...")
    save_metrics(
        accuracy,
        report,
        json_path="metrics/metrics.json",
        txt_path="metrics/metrics.txt"
    )

    print("\nDone!")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/processed/iris.csv")
    parser.add_argument("--model_path", default="models/model.pkl")
    parser.add_argument("--model_type", default="knn", choices=["knn", "logistic"])
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        model_path=args.model_path,
        model_type=args.model_type,
        k=args.k
    )
