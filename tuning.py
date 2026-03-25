"""
tuning.py
=========
Hyperparameter-Tuning für XGBoost via RandomizedSearchCV.
[Verbesserung 2]

Verwendung:
  DO_TUNING = True  in prediction_ohne_xG.py / prediction_mit_xG.py setzen,
  dann wird tune_xgboost() automatisch aufgerufen.
"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# SUCHRAUM
# ─────────────────────────────────────────────────────────────────────────────
PARAM_DIST = {
    "clf__n_estimators":      [100, 200, 300, 400],
    "clf__max_depth":         [3, 4, 5, 6, 7],
    "clf__learning_rate":     [0.03, 0.05, 0.08, 0.1, 0.15],
    "clf__subsample":         [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "clf__min_child_weight":  [1, 3, 5],
}


def tune_xgboost(X_train: np.ndarray, y_train_enc: np.ndarray,
                 cv: int = 3, n_iter: int = 30, random_state: int = 42):
    """
    Führt RandomizedSearchCV für XGBoost durch.

    Parameter:
        X_train      : Feature-Matrix (bereits skaliert oder roh – Pipeline skaliert intern)
        y_train_enc  : Label-encodierte Zielwerte (int)
        cv           : Anzahl Cross-Validation-Folds
        n_iter       : Anzahl der getesteten Parameterkombinationen
        random_state : Seed für Reproduzierbarkeit

    Rückgabe:
        Bestes Pipeline-Objekt (StandardScaler + optimierter XGBClassifier)
    """
    print(f"\n🔍 [Verbesserung 2] Hyperparameter-Tuning gestartet (n_iter={n_iter}, cv={cv}) ...")

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        ))
    ])

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train_enc)

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"\n✅ [Verbesserung 2] Beste Parameter gefunden:")
    for k, v in best_params.items():
        print(f"   {k:25} = {v}")
    print(f"   CV-Accuracy (beste): {search.best_score_:.4f} ({search.best_score_*100:.2f}%)")

    return search.best_estimator_
