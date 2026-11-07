from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import calculate_metrics


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET_COLUMN = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Dry Bean Model Lab",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading saved classifiers...")
def load_pipeline(path_string: str, modified_ns: int, file_size: int):
    """Cache a saved pipeline; file metadata invalidates cache after replacement."""
    del modified_ns, file_size  # used as cache-key values, not by joblib
    return joblib.load(path_string)


def available_pipelines():
    """Return existing saved artifacts and names that still need training."""
    existing = {}
    missing = []
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.is_file():
            stat = path.stat()
            try:
                existing[display_name] = load_pipeline(
                    str(path), stat.st_mtime_ns, stat.st_size
                )
            except Exception as exc:
                st.error(f"Could not load **{display_name}**: {exc}")
        else:
            missing.append(display_name)
    return existing, missing


def validate_upload(data: pd.DataFrame, required_features: list[str], known_classes):
    """Check upload structure and values before model inference."""
    if data.empty:
        return None, None, "The CSV has headers but no data rows."
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        return None, None, f"Duplicate column names found: {duplicates}"
    if TARGET_COLUMN not in data.columns:
        return None, None, f"Include the target column `{TARGET_COLUMN}` in the CSV."

    missing = [name for name in required_features if name not in data.columns]
    if missing:
        return None, None, f"Missing required feature columns: {', '.join(missing)}"

    # Ignore extra columns, but make the behavior visible to the user.
    extra = [
        name for name in data.columns
        if name not in required_features and name != TARGET_COLUMN
    ]
    if extra:
        st.info("Extra columns will be ignored: " + ", ".join(extra))

    features = data.loc[:, required_features].copy()
    for column in required_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    invalid_columns = features.columns[features.isna().any()].tolist()
    if invalid_columns:
        return (
            None,
            None,
            "Input features contain missing or non-numeric values in: "
            + ", ".join(invalid_columns),
        )

    labels = data[TARGET_COLUMN]
    if labels.isna().any():
        return None, None, f"The `{TARGET_COLUMN}` column contains missing labels."

    known = set(known_classes)
    observed = set(labels.unique())
    unknown = sorted(str(value) for value in observed - known)
    if unknown:
        return None, None, "Unknown target labels: " + ", ".join(unknown)

    return features, labels, None


def score_pipeline(pipeline, features, labels):
    """Predict and calculate requested metrics without fitting the pipeline."""
    predictions = pipeline.predict(features)
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "This saved pipeline does not provide predict_proba; AUC cannot be calculated."
        )
    probabilities = pipeline.predict_proba(features)
    class_labels = pipeline.classes_
    scores = calculate_metrics(labels, predictions, probabilities, class_labels)
    return scores, predictions, class_labels


def show_confusion_matrix(labels, predictions, class_labels):
    """Draw a labeled confusion matrix for the selected model."""
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="crest",
        linewidths=0.5,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
    )
    axis.set_title("Predicted vs actual bean variety")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def main():
    st.title("Dry Bean Model Lab")
    st.write(
        "Compare the five saved classifiers on labeled holdout data. "
        "The app predicts and evaluates uploaded rows; it does not train models."
    )

    with st.sidebar:
        st.header("Evaluation setup")
        st.markdown(
            "**Dataset:** UCI Dry Bean  \n"
            "**Target:** `Class`  \n"
            "**Scores:** macro Precision, Recall, F1 and OvR macro AUC"
        )
        st.caption("Upload a test CSV with all model inputs and the true Class label.")

    pipelines, missing_models = available_pipelines()
    if missing_models:
        st.warning(
            "Some saved models are not present yet: " + ", ".join(missing_models)
        )
    if not pipelines:
        st.error(
            "No usable model artifacts were found. Run `train_models.py` after "
            "completing the model setup, and keep the five `.joblib` files in `model/`."
        )
        st.stop()

    upload = st.file_uploader(
        "Choose a labeled test-data CSV",
        type=["csv"],
        help="Use held-out rows only. The CSV needs all input columns and the Class target.",
    )
    if upload is None:
        st.info("Upload a labeled test CSV to see model results.")
        st.stop()

    try:
        uploaded_data = pd.read_csv(upload)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"The uploaded file could not be read as a CSV: {exc}")
        st.stop()

    selected_name = st.selectbox("Inspect model", list(pipelines))
    selected_pipeline = pipelines[selected_name]
    if not hasattr(selected_pipeline, "feature_names_in_"):
        st.error(
            f"{selected_name} does not record its training feature names. "
            "Retrain and save the full preprocessing Pipeline."
        )
        st.stop()

    feature_names = list(selected_pipeline.feature_names_in_)
    X, y_true, validation_error = validate_upload(
        uploaded_data, feature_names, selected_pipeline.classes_
    )
    if validation_error:
        st.error(validation_error)
        st.stop()

    st.caption(f"Evaluating {len(uploaded_data):,} labeled rows.")
    with st.expander("Preview uploaded data"):
        st.dataframe(uploaded_data.head(10), use_container_width=True)

    # Score every available artifact so the comparison stays visible together.
    result_rows = []
    predictions_by_model = {}
    errors_by_model = {}
    for name, pipeline in pipelines.items():
        try:
            scores, predictions, classes = score_pipeline(pipeline, X, y_true)
            result_rows.append({"Model": name, **scores})
            predictions_by_model[name] = (predictions, classes)
        except Exception as exc:
            errors_by_model[name] = str(exc)

    if not result_rows:
        st.error("None of the available models could evaluate this CSV.")
        for name, message in errors_by_model.items():
            st.write(f"**{name}:** {message}")
        st.stop()

    st.subheader("Model comparison")
    comparison = pd.DataFrame(result_rows).set_index("Model")
    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True,
    )
    if errors_by_model:
        with st.expander("Models that could not be scored"):
            for name, message in errors_by_model.items():
                st.write(f"**{name}:** {message}")

    if selected_name not in predictions_by_model:
        st.error(f"Could not evaluate {selected_name}; see the comparison details above.")
        st.stop()

    selected_predictions, selected_classes = predictions_by_model[selected_name]
    st.subheader(f"{selected_name}: detailed results")
    st.dataframe(
        comparison.loc[[selected_name]].style.format("{:.4f}"),
        use_container_width=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("#### Confusion matrix")
        show_confusion_matrix(y_true, selected_predictions, selected_classes)
    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_true,
            selected_predictions,
            labels=selected_classes,
            target_names=[str(value) for value in selected_classes],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().style.format(precision=3),
            use_container_width=True,
        )

    st.caption(
        "Metrics are calculated from the uploaded labels. Keep this file separate "
        "from training and model selection."
    )


if __name__ == "__main__":
    main()

