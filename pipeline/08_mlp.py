import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

from config import TRAIN_FILE, TEST_FILE, FEATURES
from evaluate import evaluate

# ============================================================
# Load train and test data
# ============================================================
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X_train = train.drop(columns=["Winner_is_p1"])
y_train = train["Winner_is_p1"]

X_test = test.drop(columns=["Winner_is_p1"])
y_test = test["Winner_is_p1"]

X_train = X_train[FEATURES]
X_test = X_test[FEATURES]

# ============================================================
#  Hyperparameter search space
# ============================================================
PARAM_GRID = {
    "model__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
    "model__alpha": [0.0001, 0.001, 0.01, 0.05],
    "model__learning_rate_init": [0.0005, 0.001, 0.005],
    "model__activation": ["relu","tanh"],
}

# ============================================================
# Pipeline
# ============================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        solver="adam",
        learning_rate="adaptive",
        early_stopping=True,
        max_iter=1000,
        random_state=42,
        verbose=False,
    )),
])

# ============================================================
# Nested cross validation
# ============================================================
inner_cv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=PARAM_GRID,
    n_iter=5,
    scoring="accuracy",
    cv=inner_cv,
    n_jobs=-1,
    random_state=42,
)

outer_cv = TimeSeriesSplit(n_splits=5)

scorings = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}

nested_scores = cross_validate(
    search,
    X_train,
    y_train,
    cv=outer_cv,
    scoring=scorings,
    n_jobs=-1,
)

# Print results
print("\n" + "=" * 50)
print("NESTED CROSS VALIDATION RESULTS")
print("=" * 50)
print("Accuracy:", np.mean(nested_scores["test_accuracy"]))
print("Precision:", np.mean(nested_scores["test_precision"]))
print("Recall:", np.mean(nested_scores["test_recall"]))
print("ROC AUC:", np.mean(nested_scores["test_roc_auc"]))
print("Brier Score:", -np.mean(nested_scores["test_brier"]))

# =========================================================
#                    MODEL TRAINING
# =========================================================
# Final RandomCV on full training set
tscv = TimeSeriesSplit(n_splits=5)
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=PARAM_GRID,
    n_iter=10,
    scoring="accuracy",
    cv=tscv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

# Predict
y_pred = search.predict(X_test)
y_pred_proba = search.predict_proba(X_test)[:, 1]

# Print results
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print("Hyperparameters:", search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("Brier Score:", brier_score_loss(y_test, y_pred_proba))

# ============================================================
# Save evaluation outputs
# ============================================================
evaluate(
    model_name="MLP",
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    hyperparameters=search.best_params_,
    feature_names=None,
    importances=None,
)