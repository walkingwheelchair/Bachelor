"""
prediction_ohne_xG.py
=====================
Vorhersage von Bundesliga-Spielergebnissen OHNE Expected-Goals-Features.
Verwendet zwei Modelle:
  1. Logistic Regression
  2. Random Forest
Trainings-Saisons: 2016/17 – 2022/23
Test-Saisons:      2023/24 – 2025/26
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from utils import prepare_dataset, BASE_FEATURES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SEASONS_END = "2022/23"   # alles bis einschl. diese Saison = Training
LABEL_ORDER       = ["H", "D", "A"]
RESULTS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ergebnisse")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(df: pd.DataFrame, train_end: str, features: list[str]):
    """Splittet den DataFrame in Train/Test anhand der Saison."""
    seasons = sorted(df["Season"].unique())
    train_seasons = [s for s in seasons if s <= train_end]
    test_seasons  = [s for s in seasons if s >  train_end]

    train = df[df["Season"].isin(train_seasons)]
    test  = df[df["Season"].isin(test_seasons)]

    X_train = train[features].values
    y_train = train["FTR"].values
    X_test  = test[features].values
    y_test  = test["FTR"].values

    return X_train, y_train, X_test, y_test, train_seasons, test_seasons


def evaluate_model(name: str, y_test, y_pred, y_proba) -> dict:
    """Berechnet Accuracy, Log-Loss und Confusion-Matrix für ein Modell."""
    acc  = accuracy_score(y_test, y_pred)
    ll   = log_loss(y_test, y_proba, labels=LABEL_ORDER)
    cm   = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    rep  = classification_report(y_test, y_pred, labels=LABEL_ORDER,
                                  target_names=["Heimsieg (H)", "Unentschieden (D)", "Auswärtssieg (A)"])

    print(f"\n{'='*60}")
    print(f"  Modell: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Log-Loss  : {ll:.4f}")
    print(f"\n{rep}")

    return {"name": name, "accuracy": acc, "log_loss": ll, "cm": cm, "y_pred": y_pred, "y_proba": y_proba}


def plot_confusion_matrix(results: list[dict], suffix: str = ""):
    """Erstellt Confusion-Matrix-Plots für alle Modelle nebeneinander."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=["Heim (H)", "Unentsch. (D)", "Auswärts (A)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']:.3f}", fontsize=12, fontweight="bold")

    fig.suptitle(f"Confusion Matrices – OHNE xG{suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_ohne_xG{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


def plot_feature_importance(model_xgb, features: list[str], suffix: str = ""):
    """Plottet Feature-Importance des XGBoost Modells."""
    importances = model_xgb.named_steps["clf"].feature_importances_
    indices     = np.argsort(importances)[::-1][:20]  # Nur Top 20
    sorted_feat = [features[i] for i in indices]
    sorted_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color="steelblue")
    ax.set_xlabel("Wichtigkeit", fontsize=12)
    ax.set_title("Feature Importance – XGBoost (OHNE xG) - Top 20", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"feature_importance_ohne_xG{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  BUNDESLIGA VORHERSAGE – OHNE xG")
    print("=" * 60)

    # 1. Daten laden
    df, features = prepare_dataset(include_xg=False)

    # 2. Train/Test Split
    X_train, y_train, X_test, y_test, train_s, test_s = split_train_test(
        df, TRAIN_SEASONS_END, features
    )
    print(f"\n📅 Training-Saisons  : {train_s}")
    print(f"📅 Test-Saisons      : {test_s}")
    print(f"   Train-Spiele: {len(X_train)} | Test-Spiele: {len(X_test)}")

    # 3. Label-Verteilung
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"\n📊 Test-Verteilung: {dict(zip(unique, counts))}")

    # 4. Modelle definieren
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, C=1.0, random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                random_state=42, n_jobs=-1, eval_metric="mlogloss"
            ))
        ]),
    }

    # 5. Trainieren & Evaluieren
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    all_results = []
    trained_models = {}
    for name, model in models.items():
        print(f"\n🚀 Trainiere: {name} ...")
        if "XGBoost" in name:
            model.fit(X_train, y_train_enc)
            y_pred_num = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_num)
            y_proba = model.predict_proba(X_test)
            result = evaluate_model(name, y_test, y_pred, y_proba)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            result = evaluate_model(name, y_test, y_pred, y_proba)
            
        result["y_true"] = y_test   # <-- für Vergleichsskript
        all_results.append(result)
        trained_models[name] = model

    # 6. Visualisierungen
    print("\n📈 Erstelle Plots ...")
    plot_confusion_matrix(all_results)
    plot_feature_importance(trained_models["XGBoost"], features)

    # 7. Ergebnisse als CSV speichern
    summary = pd.DataFrame([{
        "Modell":    r["name"],
        "Accuracy":  round(r["accuracy"], 4),
        "Log_Loss":  round(r["log_loss"], 4),
        "Variante":  "ohne_xG"
    } for r in all_results])
    csv_path = os.path.join(RESULTS_DIR, "ergebnisse_ohne_xG.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  💾 Gespeichert: {csv_path}")

    print("\n✅ Vorhersage (OHNE xG) abgeschlossen.")
    return all_results, trained_models


if __name__ == "__main__":
    main()
