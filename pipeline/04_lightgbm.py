import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# Define the features to use
FEATURES = [
    'ht_diff',
    'age_diff',
    'rank_diff',
    'rank_points_diff',
    'ace_rate_10_diff',
    'df_rate_10_diff',
    'bp_save_rate_10_diff',
    'serve_win_rate_10_diff',
    '1stWon_rate_10_diff',
    '2ndWon_rate_10_diff',
    'h2h_diff',
    'win_rate_10_diff',
    'elo_before_diff',
    'elo_surface_before_diff',
    'elo_surface_blended_before_diff',
    'surface_Carpet',
    'surface_Clay',
    'surface_Grass',
    'surface_Hard',
]

# Load the train and test set
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Test train/test split
X_train = train.drop(columns=['Winner_is_p1'])
y_train = train['Winner_is_p1']
X_test = test.drop(columns=['Winner_is_p1'])
y_test = test['Winner_is_p1']

# Select features
X_train = X_train[FEATURES]
X_test = X_test[FEATURES]


# Hyperparameter search space for random search
PARAM_GRID = {
    "n_estimators": [120, 150, 180, 200],
    "learning_rate": [0.05, 0.06, 0.07],
    "num_leaves": [15, 31],
    "min_child_samples": [40, 60, 80],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# =========================================================
#                    NESTED CROSS VALIDATION
# =========================================================

# Nested cross validation
model = lgb.LGBMClassifier(
    random_state=42,
    verbosity=-1
)

# Inner cross validation for hyperparameter selection
inner_cv = TimeSeriesSplit(n_splits=5)

# Random search cross validation
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_GRID,
    n_iter=40,
    scoring="accuracy",
    cv=inner_cv,
    verbose=1,
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
# print("Accuracy:", np.mean(nested_scores["test_accuracy"]))
# print("Precision:", np.mean(nested_scores["test_precision"]))
# print("Recall:", np.mean(nested_scores["test_recall"]))
# print("ROC AUC:", np.mean(nested_scores["test_roc_auc"]))

# Initialize LightGBM classifier
model = lgb.LGBMClassifier(
    random_state=42,
    max_depth = -1,
    verbosity=-1
)

# =========================================================
#                    MODEL TRAINING
# =========================================================

# Create time series aware cross validation
tscv = TimeSeriesSplit(n_splits=5)

# Run random search cv and fit the best model
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_GRID,
    n_iter=100,
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

# {'subsample': 0.8, 'num_leaves': 15, 'n_estimators': 200, 'min_child_samples': 40, 'max_depth': 3, 'learning_rate': 0.07, 'colsample_bytree': 0.8}

# Print results
print(search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))