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
from sklearn.metrics import ConfusionMatrixDisplay

from prediction_ohne_xG import main as run_ohne_xG, split_train_test as split_ohne, TRAIN_SEASONS_END as TRAIN_END
from prediction_mit_xG   import main as run_mit_xG

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Ergebnisse")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# VERGLEICHS-PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_logloss(all_results: list[dict]):
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
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

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

    # Plots & Tabelle
    print("\n\n📊 Erstelle Vergleichs-Visualisierungen ...")
    plot_accuracy_logloss(all_results)
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
