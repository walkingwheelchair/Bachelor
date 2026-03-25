"""
vergleich.py
============
Vergleicht ALLE 4 Modelle (2x ohne xG, 2x mit xG) in einer Übersicht.
Erstellt:
  - Balkendiagramm Accuracy & Log-Loss aller 4 Modelle
  - Alle 4 Confusion Matrices nebeneinander
  - Zusammenfassende Tabelle
  - Interpretation / Fazit
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, log_loss

from prediction_ohne_xG import main as run_ohne_xG, split_train_test as split_ohne, TRAIN_SEASONS_END as TRAIN_END
from prediction_mit_xG   import main as run_mit_xG
from utils import prepare_dataset, BASE_FEATURES

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ergebnisse")
os.makedirs(RESULTS_DIR, exist_ok=True)
LABEL_ORDER = ["H", "D", "A"]


# ─────────────────────────────────────────────────────────────────────────────
# [V4] BASELINE-VERGLEICHE
# ─────────────────────────────────────────────────────────────────────────────

def compute_baselines(y_test, df_test=None):
    """
    Berechnet zwei Baselines:
    1. Immer Heimsieg (H)
    2. Wettquoten-Baseline (Bet365, falls vorhanden)
    """
    print("\n✅ [Verbesserung 4] Baseline-Vergleiche aktiv")
    baselines = {}

    # --- Baseline 1: Immer Heimsieg ---
    y_home = np.array(["H"] * len(y_test))
    acc_home = np.mean(y_home == y_test)
    # Log-Loss: P(H)=1, P(D)=0, P(A)=0  → verwende kleinen Clip gegen -inf
    proba_home = np.tile([1 - 1e-7, 5e-8, 5e-8], (len(y_test), 1))
    ll_home = log_loss(y_test, proba_home, labels=LABEL_ORDER)
    baselines["Immer Heimsieg"] = {"accuracy": acc_home, "log_loss": ll_home}
    print(f"   Baseline 'Immer Heimsieg': Accuracy={acc_home:.4f}, Log-Loss={ll_home:.4f}")

    # --- Baseline 2: Wettquoten (Bet365) ---
    if df_test is not None and all(c in df_test.columns for c in ["B365H", "B365D", "B365A"]):
        try:
            b365 = df_test[["B365H", "B365D", "B365A"]].apply(pd.to_numeric, errors="coerce").dropna()
            idx  = b365.index
            # Implizite Wahrscheinlichkeiten (1/Quote), dann normalisieren (Vig entfernen)
            p_raw = 1.0 / b365.values
            p_norm = p_raw / p_raw.sum(axis=1, keepdims=True)  # [P(H), P(D), P(A)]
            y_bet = np.array([LABEL_ORDER[i] for i in np.argmax(p_norm, axis=1)])
            y_true_bet = np.array(y_test)[idx - idx.min()]
            acc_bet = np.mean(y_bet == y_true_bet)
            ll_bet  = log_loss(y_true_bet, p_norm, labels=LABEL_ORDER)
            baselines["Wettquoten (Bet365)"] = {"accuracy": acc_bet, "log_loss": ll_bet}
            print(f"   Baseline 'Wettquoten (Bet365)': Accuracy={acc_bet:.4f}, Log-Loss={ll_bet:.4f}")
        except Exception as e:
            print(f"   ⚠️  Wettquoten-Baseline konnte nicht berechnet werden: {e}")
    else:
        print("   ⚠️  Keine Wettquoten-Spalten (B365H/D/A) in Testdaten – Baseline übersprungen.")

    return baselines

def plot_accuracy_logloss(all_results: list[dict], baselines: dict = None):
    """Balkendiagramm: Accuracy & Log-Loss aller 4 Modelle."""
    names     = [r["name"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]
    loglosses  = [r["log_loss"] for r in all_results]
    colors     = ["steelblue", "steelblue", "tomato", "tomato"]
    hatches    = ["", "///", "", "///"]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Accuracy ---
    bars1 = ax1.bar(x, accuracies, width=0.6, color=colors)
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy im Vergleich\n(höher = besser)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="Zufalls-Baseline (1/3)")
    # [V4] Weitere Baselines als gestrichelte Linien
    if baselines:
        bl_colors = ["darkorange", "purple"]
        for (bl_name, bl_vals), bl_col in zip(baselines.items(), bl_colors):
            ax1.axhline(y=bl_vals["accuracy"], color=bl_col, linestyle="--",
                        linewidth=1.2, label=f"Baseline: {bl_name}")
    ax1.legend(fontsize=9)
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Log-Loss ---
    bars2 = ax2.bar(x, loglosses, width=0.6, color=colors)
    for bar, hatch in zip(bars2, hatches):
        bar.set_hatch(hatch)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("Log-Loss", fontsize=12)
    ax2.set_title("Log-Loss im Vergleich\n(niedriger = besser)", fontsize=12, fontweight="bold")
    for bar, val in zip(bars2, loglosses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Legende Farben
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="steelblue", label="OHNE xG"),
        Patch(color="tomato",    label="MIT xG"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "vergleich_accuracy_logloss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


def plot_all_confusion_matrices(all_results: list[dict]):
    """Alle 4 Confusion Matrices in einem 2x2-Grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    cmaps = ["Blues", "Blues", "Greens", "Greens"]

    for ax, res, cmap in zip(axes, all_results, cmaps):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["cm"],
            display_labels=["Heim (H)", "Unentsch. (D)", "Auswärts (A)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap=cmap)
        variante = "OHNE xG" if "ohne" in res.get("variante", "ohne") else "MIT xG"
        ax.set_title(f"{res['name']} [{variante}]\nAccuracy: {res['accuracy']:.3f}", fontsize=11, fontweight="bold")

    fig.suptitle("Alle 4 Modelle – Confusion Matrices im Vergleich",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "vergleich_alle_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


def plot_radar_comparison(all_results: list[dict]):
    """
    Radar-/Spider-Chart: zeigt Precision H/D/A für alle 4 Modelle.
    """
    from sklearn.metrics import precision_score

    categories = ["Precision H", "Precision D", "Precision A", "Recall H", "Recall A"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # schließe Plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_list = ["steelblue", "cornflowerblue", "tomato", "salmon"]
    labels = [f"{r['name']} [{'MIT' if r.get('variante')=='mit' else 'OHNE'} xG]" for r in all_results]

    for res, color, label in zip(all_results, colors_list, labels):
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, _, _ = precision_recall_fscore_support(
            res["y_true"], res["y_pred"], labels=["H", "D", "A"], zero_division=0
        )
        values = [prec[0], prec[1], prec[2], rec[0], rec[2]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_title("Precision & Recall – Radar-Vergleich", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "vergleich_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Gespeichert: {path}")


def print_summary_table(all_results: list[dict]):
    """Gibt eine formatierte Zusammenfassungstabelle aus."""
    print("\n" + "=" * 70)
    print("  VERGLEICHS-ZUSAMMENFASSUNG")
    print("=" * 70)
    print(f"{'Modell':<35} {'Variante':<12} {'Accuracy':>10} {'Log-Loss':>10}")
    print("-" * 70)
    for r in all_results:
        variante = "MIT xG" if r.get("variante") == "mit" else "OHNE xG"
        print(f"{r['name']:<35} {variante:<12} {r['accuracy']:>10.4f} {r['log_loss']:>10.4f}")
    print("=" * 70)

    # Beste Modelle
    best_acc  = max(all_results, key=lambda x: x["accuracy"])
    best_loss = min(all_results, key=lambda x: x["log_loss"])
    variante_acc  = "MIT xG" if best_acc.get("variante")  == "mit" else "OHNE xG"
    variante_loss = "MIT xG" if best_loss.get("variante") == "mit" else "OHNE xG"
    print(f"\n🏆 Beste Accuracy : {best_acc['name']} [{variante_acc}]  → {best_acc['accuracy']:.4f}")
    print(f"🏆 Bester Log-Loss: {best_loss['name']} [{variante_loss}] → {best_loss['log_loss']:.4f}")

    # xG-Effekt
    ohne = [r for r in all_results if r.get("variante") != "mit"]
    mit  = [r for r in all_results if r.get("variante") == "mit"]
    for o, m in zip(ohne, mit):
        diff = m["accuracy"] - o["accuracy"]
        sign = "+" if diff >= 0 else ""
        effect = "✅ xG verbessert" if diff > 0 else ("⚠️  xG verschlechtert" if diff < 0 else "➡️  kein Unterschied")
        print(f"\n📊 {o['name']}: {effect} das Modell ({sign}{diff*100:.2f}% Accuracy)")


# ─────────────────────────────────────────────────────────────────────────────
# [V3] McNEMAR-TEST
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(result_ohne: dict, result_mit: dict):
    """
    Führt den McNemar-Test durch, um zu prüfen ob der Unterschied
    zwischen OHNE-xG und MIT-xG Modell statistisch signifikant ist.
    """
    print("\n✅ [Verbesserung 3] McNemar-Test aktiv")
    df_o = result_ohne["df_test"].copy()
    df_o["pred_o"] = result_ohne["y_pred"]
    df_o["true_o"] = result_ohne["y_true"]
    
    df_m = result_mit["df_test"].copy()
    df_m["pred_m"] = result_mit["y_pred"]
    df_m["true_m"] = result_mit["y_true"]

    merged = df_o.merge(df_m, on=["Date", "HomeTeam", "AwayTeam"], suffixes=('_o', '_m'))

    correct_o = (merged["pred_o"] == merged["true_o"])
    correct_m = (merged["pred_m"] == merged["true_m"])

    b = np.sum( correct_o & ~correct_m)  # nur OHNE korrekt
    c = np.sum(~correct_o &  correct_m)  # nur MIT korrekt

    name = result_ohne["name"]
    print(f"\n  McNemar-Test: {name} (OHNE vs. MIT xG)")
    print(f"    Korrekt nur OHNE xG: {b} Spiele")
    print(f"    Korrekt nur MIT xG:  {c} Spiele")

    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        table = np.array([[0, b], [c, 0]])
        result = sm_mcnemar(table, exact=False, correction=True)
        chi2  = result.statistic
        p_val = result.pvalue
    except ImportError:
        from scipy.stats import chi2 as scipy_chi2
        chi2_val = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
        p_val = 1 - scipy_chi2.cdf(chi2_val, df=1)
        chi2  = chi2_val

    sig = "statistisch signifikant (p < 0.05)" if p_val < 0.05 else "NICHT signifikant"
    print(f"    Chi²: {chi2:.2f}  |  p-Wert: {p_val:.3f}")
    print(f"    → Unterschied ist [{sig}]")



def main():
    print("\n" + "=" * 60)
    print("  BUNDESLIGA VORHERSAGE – VOLLSTÄNDIGER VERGLEICH")
    print("=" * 60)

    # Beide Varianten ausführen
    print("\n\n▶ Starte Variante OHNE xG ...")
    results_ohne, _ = run_ohne_xG()

    print("\n\n▶ Starte Variante MIT xG ...")
    results_mit,  _ = run_mit_xG()

    # Varianten-Label hinzufügen
    for r in results_ohne:
        r["variante"] = "ohne"
    for r in results_mit:
        r["variante"] = "mit"

    all_results = results_ohne + results_mit

    # [V3] McNemar-Test für beide Modellpaare
    ohne_list = [r for r in all_results if r.get("variante") == "ohne"]
    mit_list  = [r for r in all_results if r.get("variante") == "mit"]
    for o, m in zip(ohne_list, mit_list):
        mcnemar_test(o, m)

    # [V4] Baselines berechnen (braucht Testdaten mit Wettquoten-Spalten)
    try:
        df_all, _ = prepare_dataset(include_xg=False)
        test_seasons = [s for s in sorted(df_all["Season"].unique()) if s > TRAIN_END]
        df_test_raw  = df_all[df_all["Season"].isin(test_seasons)]
        y_test_raw   = df_test_raw["FTR"].values
        baselines    = compute_baselines(y_test_raw, df_test_raw)
    except Exception as e:
        print(f"\n⚠️  Baseline-Berechnung fehlgeschlagen: {e}")
        baselines = {}

    # Plots & Tabelle
    print("\n\n📊 Erstelle Vergleichs-Visualisierungen ...")
    plot_accuracy_logloss(all_results, baselines=baselines)
    plot_all_confusion_matrices(all_results)
    plot_radar_comparison(all_results)
    print_summary_table(all_results)

    # Gesamt-CSV
    summary = pd.DataFrame([{
        "Modell":    r["name"],
        "Variante":  "MIT xG" if r["variante"] == "mit" else "OHNE xG",
        "Accuracy":  round(r["accuracy"], 4),
        "Log_Loss":  round(r["log_loss"],  4),
    } for r in all_results])
    csv_path = os.path.join(RESULTS_DIR, "vergleich_gesamt.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  💾 Gesamt-CSV: {csv_path}")

    print("\n✅ Vollständiger Vergleich abgeschlossen!")
    print(f"   Alle Ergebnisse in: {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
