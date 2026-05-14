import json
import os
import csv
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)



def evaluate(model_name, y_test, y_pred, y_pred_proba, hyperparameters: dict, feature_names, importances):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{model_name.replace(' ', '_')}_{timestamp}"
    out_dir = os.path.join("results", folder_name)
    os.makedirs(out_dir)

    _save_metrics(y_test, y_pred, y_pred_proba, model_name, timestamp)
    _plot_confusion_matrix(y_test, y_pred, model_name, out_dir)
 
    if hyperparameters is not None:
        _save_hyperparameters(hyperparameters, out_dir)
 
    if feature_names is not None and importances is not None:
        _plot_feature_importance(feature_names, importances, model_name, 15, out_dir)



def _save_metrics(y_test, y_pred, y_pred_proba, model_name, timestamp):
    metrics = {
        "model":     model_name,
        "timestamp": timestamp,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_pred_proba), 4) if y_pred_proba is not None else None,
        "brier_score": round(brier_score_loss(y_test, y_pred_proba), 4) if y_pred_proba is not None else None,
    }

    csv_path = 'results/metrics.csv'
    file_exists = os.path.isfile(csv_path)
 
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    print(f"\n  Metrics saved to {csv_path}")


def _plot_confusion_matrix(y_test, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_display.plot()
    plt.title(f"{model_name} — Confusion Matrix")
 
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
 
    print(f"\n  Confusion matrix saved to: {cm_path}")


def _save_hyperparameters(hyperparameters: dict, out_dir):
    json_path = os.path.join(out_dir, "hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
 
    print(f"\n  Hyperparameters saved to: {json_path}")


def _plot_feature_importance(feature_names, importances, model_name, top_n, out_dir):
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)
 
    top_features = importance_df.head(top_n)
 
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name} — Feature Importance (Top {top_n})")
    plt.xlabel("Importance")
    plt.tight_layout()
 
    importance_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(importance_path, bbox_inches="tight")
    plt.close()
 
    print(f"\n  Feature importance saved to: {importance_path}")