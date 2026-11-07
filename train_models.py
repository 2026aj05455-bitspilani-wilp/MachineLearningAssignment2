from pathlib import Path

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from metrics_utils import calculate_metrics


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "training_data.csv"
TEST_PATH = ROOT / "test_data.csv"
TARGET = "Class"
MODEL_DIR = ROOT / "model"


def make_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("classifier", DecisionTreeClassifier(max_depth=12, random_state=42))
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[("classifier", GaussianNB())]
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
                ),
            ]
        ),
    }


def main():
    training = pd.read_csv(TRAIN_PATH)
    testing = pd.read_csv(TEST_PATH)
    X_train = training.drop(columns=[TARGET])
    y_train = training[TARGET]
    X_test = testing.drop(columns=[TARGET])
    y_test = testing[TARGET]

    pipelines = make_models()
    MODEL_DIR.mkdir(exist_ok=True)
    rows = []

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(f"{name} must provide class probabilities for AUC.")
        probabilities = pipeline.predict_proba(X_test)
        scores = calculate_metrics(
            y_test, predictions, probabilities, pipeline.classes_
        )
        rows.append({"Model": name, **scores})

        # These names must match MODEL_FILES in app.py.
        artifact_name = {
            "Logistic Regression": "logistic_regression.joblib",
            "Decision Tree": "decision_tree.joblib",
            "KNN": "knn.joblib",
            "Naive Bayes": "naive_bayes.joblib",
            "Support Vector Machine": "support_vector_machine.joblib",
        }[name]
        joblib.dump(pipeline, MODEL_DIR / artifact_name)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "metrics_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

