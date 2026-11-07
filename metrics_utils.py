"""Metric helper for the assignment scaffold; read the choices before use."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, y_pred, y_score, class_labels):
    """Return the six requested metrics for a multiclass classification run.

    y_score must contain probabilities in the same class order as class_labels.
    This helper uses macro averaging for precision, recall, F1, and OvR AUC.
    """
    y_true_binary = label_binarize(y_true, classes=class_labels)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC (OvR macro)": roc_auc_score(
            y_true_binary, y_score, average="macro", multi_class="ovr"
        ),
        "Precision (macro)": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall (macro)": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

