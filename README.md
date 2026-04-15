# ⚽ Vorhersage von Bundesliga-Spielergebnissen – Bachelorarbeit

Dieses Repository enthält den vollständigen Code zur Bachelorarbeit:

> **„Verbessert die Integration von Expected-Goals-Daten (xG) die Vorhersagegenauigkeit eines Machine-Learning-Modells für Bundesliga-Spielergebnisse signifikant?"**

---

## 📋 Inhaltsverzeichnis

1. [Installation](#-installation)
2. [Projektstruktur](#-projektstruktur)
3. [Aufbau der Arbeit](#-aufbau-der-arbeit)
4. [Datenbasis](#-datenbasis)
5. [Pipeline im Überblick](#-pipeline-im-überblick)
6. [Technologie-Stack](#-technologie-stack)

---

## 🚀 Installation

**Python-Version:** 3.9+

```bash
# Virtuelle Umgebung erstellen (empfohlen)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib lightgbm

# Optional: Jupyter für das Notebook
pip install notebook
```

---

## 📂 Projektstruktur

```
Bachelor Arbeit/
│
├── utils.py                        # Datenpipeline & Feature Engineering
│
├── Bundesliga_Quellen/             # Spieldaten: 10 CSV-Dateien (2016/17–2025/26)
│   ├── Bundesliga 2016-2017.csv
│   ├── ...
│   └── Bundesliga 2025-2026.csv
│
├── xG_Quellen/                     # xG-Saisondaten: 10 CSV-Dateien
│   ├── xG 2016-2017.csv
│   ├── ...
│   └── xG 2025-2026.csv
│
├── xx_Notebook/
│   ├── notebook.ipynb              # Vollständige ML-Pipeline (31 Zellen)
│   ├── Features_Metriken.md        # Dokumentation aller Features & Metriken
│   └── best_model_*.pkl            # Gespeichertes bestes Modell (nach Notebook-Lauf)
│
└── ___Bibliothek/                  # LaTeX-Kapitel der Bachelorarbeit
    ├── Einleitung.tex
    ├── chapter2.tex                # Theoretische Grundlagen
    ├── chapter3.tex                # Related Work
    ├── chapter4.tex                # Methodik
    └── quellen.bib
```

---

## 📖 Aufbau der Arbeit

Die Bachelorarbeit folgt diesem Kapitelaufbau:

| Kapitel | Inhalt |
|---------|--------|
| 1 | Einleitung – Motivation, Forschungsfrage, Aufbau |
| 2 | Theoretische Grundlagen – ML-Algorithmen, Vorhersagbarkeit multivariater Ereignisse, Dimensionsreduktion, Expected Goals |
| 3 | Related Work – Stand der Forschung |
| 4 | Methodik – Experimentdesign, Data-Leakage-Prävention, Evaluationsmetriken |
| 5 | Modellierung & Implementierung – Datengrundlage, Feature Engineering, Umsetzung |
| 6 | Evaluation & Ergebnisvergleich |
| 7 | Diskussion der Ergebnisse & Limitationen |
| 8 | Fazit & Ausblick |

---

## 📊 Datenbasis

### Bundesliga-Spieldaten (`Bundesliga_Quellen/`)

10 Saisons (2016/17–2025/26), ~2.997 Spiele, 18 Teams pro Saison.
Genutzte Spalten: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HS`, `AS`, `HST`, `AST`, `HTHG`, `HTAG`.

**Zielvariable:** `FTR` – Full Time Result (`H` = Heimsieg, `D` = Unentschieden, `A` = Auswärtssieg)

### xG-Saisondaten (`xG_Quellen/`)

Quelle: [Understat.com](https://understat.com), 180 Team-Saison-Einträge.
Spalten: `team`, `matches`, `xG`, `xGA`, `xPTS` (+ abgeleitete per-Spiel-Werte).

**Data-Leakage-Prävention:** Für abgeschlossene Saisons werden die xG-Werte der Vorsaison als Prior eingesetzt (Lag-1-Strategie). Nur für die laufende Saison werden aktuelle Werte direkt genutzt.

---

## ⚙️ Pipeline im Überblick

### 1. `utils.py` – Datenpipeline

| Funktion | Beschreibung |
|----------|-------------|
| `load_bundesliga_data()` | Lädt alle 10 Saison-CSVs, normalisiert Teamnamen, parsed Daten |
| `load_xg_data()` | Lädt xG-Saisondaten, berechnet per-Spiel-Werte |
| `compute_rolling_features(df, span=10)` | Berechnet EWA-Features für Heim- und Auswärtsteam |
| `merge_xg_features(match_df, xg_df)` | Joined xG-Daten mit Lag-1-Strategie |
| `add_derived_features(df, include_xg)` | Differenz-Features, Unentschieden-Indikatoren |
| `prepare_dataset(include_xg)` | Komplettpipeline → fertiger DataFrame + Feature-Liste |

**Feature-Sets:**
- `BASE_FEATURES`: 25 Features (EWA-Formwerte + Differenz-Features, ohne xG)
- `XG_FEATURES`: 34 Features (BASE_FEATURES + 9 xG-basierte Merkmale)

**EWA-Berechnung:** Exponentially Weighted Average mit `span=10` (α ≈ 0,18), berechnet in drei Kontexten: Overall, Home Form, Away Form. Shift(1) vor der EWA-Berechnung verhindert, dass das aktuelle Spiel in seine eigenen Features einfließt.

### 2. `notebook.ipynb` – ML-Pipeline (31 Zellen)

**Schritt 1 – Daten laden & vorbereiten:** Spieldaten laden, EWA-Features berechnen, xG-Daten mergen, beide Basis-DataFrames (`df_base`, `df_with_xg`) aufbauen.

**Schritt 2 – EDA:** Verteilung der Zielvariable, Histogramme, Boxplots, Korrelationsmatrix (Heatmap), Feature-Target-Korrelationen, Stripplots mit Konfidenzintervallen.

**Schritt 3 – Datensatzvarianten:** 12 Kombinationen aus xG (mit/ohne) × Ausreißerbehandlung (Raw/IQR, Multiplikator 3,0) × Dimensionsreduktion (alle Features / starke Features / PCA auf 95 % Varianz).

**Schritt 4 – Training:** 5 Algorithmen × 12 Varianten = 60 Modellläufe. `RandomizedSearchCV` mit `StratifiedKFold` (5 Folds), Optimierungsmetrik: F1 macro. Klassen-Imbalance wird je Algorithmus unterschiedlich behandelt (class_weight / sample_weight / Oversampling beim MLP).

**Schritt 5 – Evaluation:** Konfusionsmatrizen, F1-Heatmap, Radar-Chart, Feature-Importance, Speicherung des besten Modells als `.pkl`.

---

## 🔑 Technologie-Stack

| Bibliothek | Verwendung |
|-----------|------------|
| `pandas` | Datenladen, Merging, Feature Engineering |
| `numpy` | Numerische Berechnungen, EWA-Formel |
| `scikit-learn` | Modelle, Scaler, StratifiedKFold, RandomizedSearchCV, Metriken |
| `xgboost` | XGBClassifier |
| `lightgbm` | LGBMClassifier |
| `matplotlib` / `seaborn` | Alle Visualisierungen |
| `joblib` | Modell-Persistenz |

---

## 📝 Schnellstart

```bash
# Notebook öffnen und ausführen
cd "Bachelor Arbeit/xx_Notebook"
jupyter notebook notebook.ipynb
```