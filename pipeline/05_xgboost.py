import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import (
    RandomizedSearchCV, 
    TimeSeriesSplit, 
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
import config as Config

# Load the train and test set
train = pd.read_csv(Config.TRAIN_FILE)
test = pd.read_csv(Config.TEST_FILE)

# Test train/test split
X_train = train.drop(columns=['Winner_is_p1'])
y_train = train['Winner_is_p1']
X_test = test.drop(columns=['Winner_is_p1'])
y_test = test['Winner_is_p1']

# Select features
X_train = X_train[Config.FEATURES]
X_test = X_test[Config.FEATURES]


# Hyperparameter search space for random search
PARAM_GRID = {
    "n_estimators": [100, 150, 200, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.07],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [ 0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# =========================================================
#                    NESTED CROSS VALIDATION
# =========================================================

# Nested cross validation
model = xgb.XGBClassifier(
    random_state=42,
)

# Inner cross validation for hyperparameter selection
inner_cv = TimeSeriesSplit(n_splits=5)

# Random search cross validation
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_GRID,
    n_iter=5,
    scoring="accuracy",
    cv=inner_cv,
    n_jobs=-1,
    random_state=42
)

# Outer cross validation for model evaluation
outer_cv = TimeSeriesSplit(n_splits=5)

# Scorings for outer cross validation
scorings = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score"
}

# Nested cv
nested_scores = cross_validate(
    search,
    X_train,
    y_train,
    cv=outer_cv,
    scoring=scorings,
    n_jobs=-1
)

# Print results
print("\n" + "="*50)
print("NESTED CROSS VALIDATION RESULTS")
print("="*50)
print("Accuracy:", np.mean(nested_scores["test_accuracy"]))
print("Precision:", np.mean(nested_scores["test_precision"]))
print("Recall:", np.mean(nested_scores["test_recall"]))
print("ROC AUC:", np.mean(nested_scores["test_roc_auc"]))
print("Brier Score:", -np.mean(nested_scores["test_brier"]))

# =========================================================
#                    MODEL TRAINING
# =========================================================
# Initialize xgb classifier
model = xgb.XGBClassifier(
    random_state=42,
)

# Create time series aware cross validation
tscv = TimeSeriesSplit(n_splits=5)

# Run random search cv and fit the best model
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_GRID,
    n_iter=10,
    scoring="accuracy",
    cv=tscv,
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

# best hyperparameter = {'subsample': 0.6, 'n_estimators': 500, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.03}


# 0.6530951692218687={'subsample': 1.0, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.01}