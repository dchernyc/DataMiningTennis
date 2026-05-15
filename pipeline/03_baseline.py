import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    classification_report
)
import config as Config
from evaluate import evaluate

#Load the test set
test = pd.read_csv(Config.TEST_FILE)
y_test = test["Winner_is_p1"]


#Baseline 1: predicts the player with the higher rank as the winner
y_pred1 = (test["p1_rank"] < test["p2_rank"]).astype(int)
print("Rank Baseline")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1))
print("Recall:", recall_score(y_test, y_pred1))
print("ROC AUC:", roc_auc_score(y_test, y_pred1))
print("Brier Score:", brier_score_loss(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

evaluate(
    model_name="Rank_Baseline",
    y_test=y_test,
    y_pred=y_pred1,
    y_pred_proba=None,
    hyperparameters=None,
    feature_names=None,
    importances=None
)

#Baseline 2: predicts the player with the higher elo score (before the match) as the winner
y_pred2 = (test["p1_elo_before"] > test["p2_elo_before"]).astype(int)
print("Elo Baseline")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2))
print("Recall:", recall_score(y_test, y_pred2))
print("ROC AUC:", roc_auc_score(y_test, y_pred2))
print("Brier Score:", brier_score_loss(y_test, y_pred2))
print(classification_report(y_test, y_pred2))

evaluate(
    model_name="Elo_Baseline",
    y_test=y_test,
    y_pred=y_pred2,
    y_pred_proba=None,
    hyperparameters=None,
    feature_names=None,
    importances=None
)