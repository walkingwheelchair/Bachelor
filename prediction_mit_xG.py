"""
prediction_mit_xG.py
====================
Vorhersage von Bundesliga-Spielergebnissen MIT Expected-Goals-Features.
Identische Struktur wie prediction_ohne_xG.py – einziger Unterschied:
  xG, xGA, xPTS (per Spiel) der jeweiligen Teams werden als Features ergänzt.

Modelle:
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from utils import prepare_dataset, XG_FEATURES

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SEASONS_END = "2022/23"
LABEL_ORDER       = ["H", "D", "A"]
RESULTS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ergebnisse")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN (analog zu prediction_ohne_xG.py)
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(df, train_end, features):
    seasons      = sorted(df["Season"].unique())
    train_seasons = [s for s in seasons if s <= train_end]
    test_seasons  = [s for s in seasons if s >  train_end]

    train = df[df["Season"].isin(train_seasons)]
    test  = df[df["Season"].isin(test_seasons)]

    return (train[features].values, train["FTR"].values,
            test[features].values,  test["FTR"].values,
            train_seasons, test_seasons)


def evaluate_model(name, model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll  = log_loss(y_test, y_proba, labels=LABEL_ORDER)
    cm  = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    rep = classification_report(y_test, y_pred, labels=LABEL_ORDER,
                                 target_names=["Heimsieg (H)", "Unentschieden (D)", "Auswärtssieg (A)"])

    print(f"\n{'='*60}")
    print(f"  Modell: {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Log-Loss  : {ll:.4f}")
    print(f"\n{rep}")

    return {"name": name, "accuracy": acc, "log_loss": ll, "cm": cm, "y_pred": y_pred, "y_proba": y_proba}


def plot_confusion_matrix(results, suffix=""):
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=["Heim (H)", "Unentsch. (D)", "Auswärts (A)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title(f"{res['name']}\nAccuracy: {res['accuracy']:.3f}", fontsize=12, fontweight="bold")

    fig.suptitle(f"Confusion Matrices – MIT xG{suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_mit_xG{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


def plot_feature_importance(model_rf, features, suffix=""):
    importances = model_rf.named_steps["clf"].feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [features[i] for i in indices]
    sorted_imp  = importances[indices]

    # xG-Features farblich hervorheben
    xg_feature_names = {"home_xG_per_game", "home_xGA_per_game", "home_xPTS_per_game",
                         "away_xG_per_game", "away_xGA_per_game", "away_xPTS_per_game",
                         "xG_diff", "xGA_diff"}
    colors = ["tomato" if f in xg_feature_names else "steelblue" for f in sorted_feat[::-1]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors)
    ax.set_xlabel("Wichtigkeit", fontsize=12)
    ax.set_title("Feature Importance – Random Forest (MIT xG)\n"
                 "(Rot = xG-Features, Blau = traditionelle Features)", fontsize=12, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    # Legende
    from matplotlib.patches import Patch
    legend = [Patch(color="tomato", label="xG-Features"),
              Patch(color="steelblue", label="Traditionelle Features")]
    ax.legend(handles=legend, loc="lower right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"feature_importance_mit_xG{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  BUNDESLIGA VORHERSAGE – MIT xG")
    print("=" * 60)

    # 1. Daten laden (mit xG)
    df, features = prepare_dataset(include_xg=True)

    # 2. Train/Test Split
    X_train, y_train, X_test, y_test, train_s, test_s = split_train_test(
        df, TRAIN_SEASONS_END, features
    )
    print(f"\n📅 Training-Saisons  : {train_s}")
    print(f"📅 Test-Saisons      : {test_s}")
    print(f"   Train-Spiele: {len(X_train)} | Test-Spiele: {len(X_test)}")

    # xG-Coverage prüfen
    xg_cols = ["home_xG_per_game", "away_xG_per_game"]
    xg_in_feat = [c for c in xg_cols if c in features]
    if xg_in_feat:
        coverage = df[xg_in_feat].notna().all(axis=1).mean()
        print(f"\n📊 xG-Daten Abdeckung: {coverage*100:.1f}% der Spiele haben xG-Werte")

    # 3. Modelle
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, C=1.0, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ))
        ]),
    }

    # 4. Trainieren & Evaluieren
    all_results = []
    trained_models = {}
    for name, model in models.items():
        print(f"\n🚀 Trainiere: {name} ...")
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        result["y_true"] = y_test   # <-- für Vergleichsskript
        all_results.append(result)
        trained_models[name] = model

    # 5. Visualisierungen
    print("\n📈 Erstelle Plots ...")
    plot_confusion_matrix(all_results)
    plot_feature_importance(trained_models["Random Forest"], features)

    # 6. Ergebnisse speichern
    summary = pd.DataFrame([{
        "Modell":   r["name"],
        "Accuracy": round(r["accuracy"], 4),
        "Log_Loss": round(r["log_loss"], 4),
        "Variante": "mit_xG"
    } for r in all_results])
    csv_path = os.path.join(RESULTS_DIR, "ergebnisse_mit_xG.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  💾 Gespeichert: {csv_path}")

    print("\n✅ Vorhersage (MIT xG) abgeschlossen.")
    return all_results, trained_models


if __name__ == "__main__":
    main()
