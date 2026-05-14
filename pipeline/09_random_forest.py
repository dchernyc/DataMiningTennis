import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from evaluate import evaluate

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
    "n_estimators": [800, 1000],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"]
}

# =========================================================
#                    NESTED CROSS VALIDATION
# =========================================================

# Nested cross validation
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
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
    verbose=0,
    n_jobs=-1,
    random_state=42
)

# Outer cross validation for model evaluation
outer_cv = TimeSeriesSplit(n_splits=3)

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

# Initialize Random Forest Classifier
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
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
    verbose=0,
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

# best hyperparameter = {'n_estimators': 1200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10}

evaluate(
    model_name="Random Forest",
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    hyperparameters=search.best_params_,
    feature_names=Config.FEATURES,
    importances=search.best_estimator_.feature_importances_
)