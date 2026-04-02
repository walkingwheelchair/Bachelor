# ⚽ Vorhersage von Bundesliga-Spielergebnissen – Bachelorarbeit

Dieses Repository enthält den vollständigen Code einer wissenschaftlichen Bachelorarbeit, die untersucht, ob und wie stark die Integration von **Expected Goals (xG)** die Vorhersagegenauigkeit von Bundesliga-Spielergebnissen verbessert.

---

## 📋 Inhaltsverzeichnis

1. [Installation](#-installation)
2. [Projektstruktur](#-projektstruktur)
3. [Wissenschaftliche Fragestellung](#-wissenschaftliche-fragestellung)
4. [Datenbasis](#-datenbasis)
5. [Komponenten im Detail](#-komponenten-im-detail)
6. [Ergebnisse](#-ergebnisse)
7. [Technologie-Stack](#-technologie-stack)

---

## 🚀 Installation

### Voraussetzungen

- **Python-Version:** 3.9.6 (getestet mit Python 3.9+)
- **Betriebssystem:** macOS / Linux / Windows

### Dependencies installieren

```bash
# Empfohlen: Virtuelle Umgebung erstellen
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Alle benötigten Packages installieren
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib lightgbm

# Optional: Jupyter Notebook für die Analyse
pip install notebook
```

### Benötigte Packages (requirements.txt Äquivalent)

```
pandas>=1.3.0      # Datenverarbeitung & Feature Engineering
numpy>=1.21.0      # Numerische Berechnungen (EWA, Arrays)
scikit-learn>=1.0  # ML-Modelle (LogisticRegression, RandomForest, Metriken)
xgboost>=1.5.0     # XGBoost Klassifikator (Hauptmodell)
lightgbm>=3.3.0    # LightGBM Alternative
matplotlib>=3.4.0  # Visualisierungen
seaborn>=0.11.0    # Statistische Plots, Heatmaps
joblib>=1.1.0      # Modell-Persistenz (Cache)
```

---

## 📂 Projektstruktur

```
Bachelor Arbeit/
│
├── utils.py                       # Zentrale Datenpipeline & Feature Engineering
├── predict_spiel.py               # Interaktives CLI-Tool für Live-Vorhersagen
│
├── Bundesliga_Quellen/            # Rohdaten: 10 CSV-Dateien mit Spielergebnissen
│   ├── Bundesliga 2016-2017.csv   # ~306 Spiele pro Saison
│   ├── Bundesliga 2017-2018.csv
│   ├── ...
│   └── Bundesliga 2025-2026.csv   # Laufende Saison
│
├── xG_Quellen/                    # Expected Goals Daten: 10 CSV-Dateien
│   ├── xG 2016-2017.csv           # 18 Teams × Saisonstatistiken
│   ├── xG 2017-2018.csv
│   ├── ...
│   └── xG 2025-2026.csv
│
├── xx_Notebook/
│   ├── notebook.ipynb             # Vollständige ML-Pipeline (31 Zellen)
│   │                              # Abschnitte:
│   │                              # 1. Daten laden & vorbereiten
│   │                              # 2. Data Exploration (EDA)
│   │                              # 3. Dataframe-Varianten (12 Stück)
│   │                              # 4. Model Training & Hyperparameter Tuning
│   │                              # 5. Finale Auswertung & Modell-Speicherung
│   │
│   ├── Features_Metriken.md       # Detaillierte Dokumentation aller Features
│   │                              # und Evaluationsmetriken mit Formeln
│   │
│   └── best_model_*.pkl           # Gespeicherte Modelle nach Notebook-Lauf
│
├── Ergebnisse/                    # Automatisch generierte Outputs
│   ├── confusion_ohne_xG.png      # Konfusionsmatrizen ohne xG
│   ├── confusion_mit_xG.png       # Konfusionsmatrizen mit xG
│   ├── feature_importance_*.png   # Feature-Importance (XGBoost)
│   ├── vergleich_accuracy_logloss.png
│   ├── vergleich_alle_confusion_matrices.png
│   ├── vergleich_radar.png        # Radar-Chart: Precision/Recall/F1
│   ├── ergebnisse_*.csv           # Numerische Metriken als CSV
│   └── vergleich_gesamt.csv       # Alle Modelle im Direktvergleich
│
└── model_cache/                   # (Optional) Cache für schnelles Testing
    ├── models_ohne_xG.joblib      # LR + XGBoost ohne xG (für Tests)
    ├── models_mit_xG.joblib       # LR + XGBoost mit xG (für Tests)
    ├── label_encoder.joblib       # LabelEncoder: H/D/A ↔ 0/1/2
    ├── features.joblib            # Feature-Listen beider Varianten
    └── meta.json                  # Timestamps für Cache-Invalidation
```

---

## 🔬 Wissenschaftliche Fragestellung

**Kernfrage:** Verbessert die Integration von Expected-Goals-Daten (xG) die Vorhersagegenauigkeit eines Machine-Learning-Modells für Bundesliga-Spielergebnisse signifikant?

### Zielvariable: Full Time Result (FTR)

| Kodierung | Label | Bedeutung |
|-----------|-------|-----------|
| `H` | Heimsieg | Heimteam gewinnt |
| `D` | Draw | Unentschieden |
| `A` | Auswärtssieg | Auswärtsteam gewinnt |

Für die Korrelationsanalyse wird FTR numerisch kodiert:
- H → **+1** (Erfolg Heimteam)
- D → **0** (Unentschieden)
- A → **−1** (Niederlage Heimteam)

### Untersuchte Modell-Varianten

Das System vergleicht systematisch **12 Kombinationen** aus:

| Achse | Optionen |
|-------|----------|
| **Algorithmus** | Logistic Regression, Random Forest, LightGBM, XGBoost, MLP (Neural Net) |
| **Feature-Set** | Ohne xG (25 Features) vs. Mit xG (34 Features) |
| **Datenvariante** | Unbereinigt vs. IQR-Ausreißer-bereinigt vs. Feature-selektiert |

---

## 📊 Datenbasis

### `Bundesliga_Quellen/` – Spielergebnisse (10 CSV-Dateien)

**Umfang:** 10 Saisons (2016/17–2025/26), ~2.997 Spiele gesamt, 18 Teams pro Saison

**Genutzte Spalten:**

| Spalte | Beschreibung | Typ |
|--------|-------------|-----|
| `Date` | Spieltag (Datum) | Date |
| `HomeTeam` / `AwayTeam` | Teamnamen | String |
| `FTHG` / `FTAG` | Tore Heim/Auswärts (Full Time Goals) | Integer |
| `FTR` | Ergebnis: H/D/A | String |
| `HS` / `AS` | Abgegebene Schüsse gesamt | Integer |
| `HST` / `AST` | Schüsse aufs Tor (Shots on Target) | Integer |
| `HTHG` / `HTAG` | Tore zur Halbzeit | Integer |

---

### `xG_Quellen/` – Expected Goals Statistiken (10 CSV-Dateien)

**Umfang:** 180 Einträge (18 Teams × 10 Saisons)

**Spalten:**

| Spalte | Beschreibung |
|--------|-------------|
| `team` | Teamname |
| `matches` | Anzahl gespielter Partien |
| `xG` | Expected Goals (Saisonsumme) |
| `xGA` | Expected Goals Against (Saisonsumme) |
| `xPTS` | Expected Points (aus xG berechnet) |

**Abgeleitete Features (per Spiel):**
- `xG_per_game` = xG / matches
- `xGA_per_game` = xGA / matches
- `xPTS_per_game` = xPTS / matches

### What is xG?

**Expected Goals (xG)** misst die Qualität von Torchancen. Jede Chance erhält einen Wert zwischen 0 und 1 basierend auf historischen Daten ähnlicher Situationen (Position, Winkel, Abstand, Körperteil, etc.).

- **xG = 1.0** bedeutet: Aus dieser Situation wird im Durchschnitt 1 Tor erwartet
- **xG = 0.2** bedeutet: 20% Chance auf Tor, im Durchschnitt 0.2 Tore

**Warum xG?** xG ist ein stabilerer Leistungsindikator als tatsächliche Tore, da es:
- Glück/Pech bei der Chancenverwertung herausrechnet
- Torwartleistungen neutralisiert
- Nachhaltige Teamstärke besser abbildet

### Data-Leakage-Prävention (Lag-1-Ansatz)

Da xG-Saisonwerte erst am Saisonende final feststehen, wird **Data Leakage** vermieden:

| Saison-Typ | xG-Quelle |
|------------|-----------|
| Abgeschlossene Saisons | Vorjahres-xG (Lag-1) als Prior |
| Aktuelle Saison | Direkte Nutzung der aktuellen Werte |

---

## ⚙️ Komponenten im Detail

### 1. `utils.py` – Datenpipeline (20.259 Bytes)

Die zentrale Bibliothek des Projekts. Wird von `notebook.ipynb` und `predict_spiel.py` importiert.

#### Konstanten

```python
BUNDESLIGA_DIR  # Pfad zu Bundesliga_Quellen/
XG_DIR          # Pfad zu xG_Quellen/
BASE_FEATURES   # 25 Basis-Features (ohne xG)
XG_FEATURES     # 34 Features (inkl. xG)
```

#### Funktionen

| Funktion | Beschreibung | Rückgabewert |
|----------|-------------|--------------|
| `normalize_team(name: str) → str` | Normalisiert Teamnamen über Mapping (z.B. "Bayern" → "Bayern Munich") | Einheitlicher Teamname |
| `load_bundesliga_data() → DataFrame` | Lädt alle 10 CSVs, fügt Season-Spalte hinzu, normalisiert Namen | ~2.997 Spiele |
| `load_xg_data() → DataFrame` | Lädt alle 10 xG-CSVs, berechnet per-Game-Werte | 180 Team-Saison-Einträge |
| `compute_rolling_features(df, span=10) → DataFrame` | Berechnet EWA-Rolling-Durchschnitte für alle Teams | DataFrame mit 20 EWA-Features |
| `merge_xg_features(match_df, xg_df) → DataFrame` | Merged xG mit Lag-1-Logik (kein Data Leakage) | DataFrame mit xG-Features |
| `add_derived_features(df, include_xg) → DataFrame` | Berechnet Differenz-Features (goal_diff, draw_tendency, etc.) | DataFrame mit abgeleiteten Features |
| `prepare_dataset(include_xg) → (DataFrame, list)` | Komplettpipeline: laden → Features → xG → cleanup | Fertiger DataFrame + Feature-Liste |

#### EWA (Exponentially Weighted Average)

Das Herzstück des Feature-Engineerings. Berechnet für jedes Spiel die historische Form beider Teams.

**Formel:**
```
EWA_neu = α × Wert_aktuell + (1 − α) × EWA_alt
mit α = 2 / (span + 1)
```

**Drei Kontexte:**
| Kontext | Beschreibung |
|---------|-------------|
| `overall` | EWA über alle Spiele (Heim + Auswärts) |
| `home_form` | EWA nur über Heimspiele des Heimteams |
| `away_form` | EWA nur über Auswärtsspiele des Auswärtsteams |

**Pro Kontext berechnete Metriken:**
`GoalsScored`, `GoalsConceded`, `Points`, `Shots`, `ShotsOnTarget`

→ **20 EWA-Features pro Spiel** (10 pro Team × 2 Teams)

#### Abgeleitete Features (Differenzen)

| Feature | Formel | Zweck |
|---------|--------|-------|
| `goal_diff_avg` | home_EWA_Goals − away_EWA_Goals | Stärkenvergleich |
| `points_diff_avg` | home_EWA_Points − away_EWA_Points | Punktvergleich |
| `abs_goal_diff` | \|goal_diff_avg\| | Unentschieden-Indikator |
| `abs_points_diff` | \|points_diff_avg\| | Unentschieden-Indikator |
| `combined_draw_tendency` | 1/(1+\|goal\|) × 1/(1+\|pts\|) | Unentschieden-Wahrscheinlichkeit |
| `xG_diff` | home_xG − away_xG | xG-Stärkenvergleich |
| `xGA_diff` | home_xGA − away_xGA | Defensiv-Vergleich |
| `abs_xG_diff` | \|xG_diff\| | xG-Unentschieden-Indikator |

---

### 2. `predict_spiel.py` – Live-Vorhersage-Tool (15.937 Bytes)

Interaktives CLI-Tool für Vorhersagen zukünftiger Bundesliga-Spiele.

#### Funktionsweise

1. **Bestes Modell laden:** Sucht `best_model_*.pkl` im `xx_Notebook/`-Ordner, lädt die Datei mit dem neuesten Timestamp
2. **Daten laden:** Bundesliga-Daten + xG-Daten via `utils.py`
3. **Team-Suche:** Fuzzy-Matching für Eingaben wie "Bayern" → "Bayern Munich"
4. **Feature-Berechnung:** Hängt Dummy-Zeile an historische Daten, berechnet EWA für bevorstehendes Spiel
5. **Vorhersage:** Transformiert Features, ruft `predict_proba()` auf, formatiert Ausgabe

#### Das eine beste Modell (aus Notebook)

Das Notebook trainiert **5 Modellklassen** (LogReg, RF, LGBM, XGBoost, MLP) auf mehreren Datensatz-Varianten und speichert das **beste einzelne Modell** basierend auf F1 macro.

| Gespeichert in | Inhalt |
|----------------|--------|
| `best_model_xgbclassifier.pkl` | XGBoost mit bester Performance (typisch) |
| `best_model_lgbmclassifier.pkl` | LightGBM (falls besser als XGBoost) |
| `best_model_randomforestclassifier.pkl` | Random Forest (falls bestes Modell) |

**Im Package enthalten:**
```python
{
    'model':       trained_model,      # Das beste trainierte Modell
    'scaler':      fitted_scaler,      # StandardScaler für Features
    'features':    feature_list,       # Liste der verwendeten Features
    'le_classes':  ['A', 'D', 'H'],    # LabelEncoder-Klassen
    'model_type':  'XGBClassifier',    # Algorithmus-Name
    'variant':     'Mit xG, IQR',      # Datensatz-Variante
    'metrics': {                       # Evaluationsmetriken auf Testset
        'f1_macro': 0.543,
        'f1_draw':  0.312,
        'accuracy': 0.567
    }
}
```

> **Hinweis:** `model_cache/` wird vom Notebook verwendet, nicht von `predict_spiel.py`. Das Tool lädt direkt aus `xx_Notebook/best_model_*.pkl`.

#### Kommandozeilen-Argumente

```bash
# Interaktiver Modus (Standard)
python3 predict_spiel.py

# Direkte Eingabe ohne Interaktion
python3 predict_spiel.py --heim "Bayern Munich" --auswaerts "Borussia Dortmund"

# Cache ignorieren und Modelle neu trainieren
python3 predict_spiel.py --no-cache

# Bestimmte Saison erzwingen
python3 predict_spiel.py --saison 2024/25
```

#### Ausgabe-Beispiel

```
=================================================================
  ⚽  VORHERSAGE:  Bayern Munich  vs.  Borussia Dortmund
=================================================================
  Modell:   XGBClassifier | Variante: Mit xG
  Metriken: F1 macro=0.543 | F1 Draw=0.312 | Acc=0.567

📊 FORM (Exponentially Weighted Averages):
                               HEIM      AUSWÄRTS
   Team                Bayern Munich  B. Dortmund
   Gew. Ø Tore erzielt         2.31          1.87
   Gew. Ø Tore kassiert        0.94          1.23
   Gew. Ø Punkte/Spiel         2.41          1.98

📈 xG-DATEN (Vorjahreswerte als Prior):
                               HEIM      AUSWÄRTS
   xG / Spiel                  2.18          1.92
   xGA / Spiel                 0.89          1.15
   xPTS / Spiel                2.35          2.01

🎯 ERWARTETE TORE (grobe Schätzung):
   Bayern Munich              2.12  |  Borussia Dortmund        1.56

=================================================================
  📋  MODELL-VORHERSAGE (XGBClassifier)
=================================================================
  Ergebnis           P(Heimsieg)   P(Unentsch.)    P(Auswärts)
  ─────────────────────────────────────────────────────────────
  🏠 Heimsieg ⚽           61.2%         19.4%         19.4%

  🏆 TIPP:  Heimsieg ⚽  (Konfidenz: 61.2%)
=================================================================
```

---

### 3. `xx_Notebook/notebook.ipynb` – ML-Pipeline (31 Zellen)

Das wissenschaftliche Kernstück – vollständige Analyse von EDA bis Modell-Evaluation.

#### Abschnitt 1: Daten laden & vorbereiten (Zelle 1-3)

- Import aller Libraries (pandas, numpy, sklearn, matplotlib, seaborn, xgboost, lightgbm)
- Laden der Spieldaten via `utils.load_bundesliga_data()`
- Berechnen der EWA-Rolling-Features
- Erstellen zweier Basis-DataFrames:
  - `df_base` (~2.955 Spiele, 25 Features ohne xG)
  - `df_with_xg` (~2.280 Spiele, 34 Features mit xG)

#### Abschnitt 2: Data Exploration / EDA (Zelle 4-13)

| Visualisierung | Zweck |
|---------------|-------|
| FTR-Balkendiagramm | Verteilung H/D/A (zeigt Heimvorteil ~45%) |
| 25 Histogramme | Verteilung jeder einzelnen Variable |
| 25 Boxplots | Ausreißer-Erkennung pro Variable |
| 25 Stripplots + Regression | Feature vs. FTR (gruppiert) |
| Korrelationsmatrix (Heatmap) | Feature-Feature-Korrelationen |
| Feature-Target-Korrelation (Barplot) | Stärkste Prädiktoren für Heimerfolg |

#### Abschnitt 3: Dataframe-Varianten erstellen (Zelle 14-17)

**12 Varianten** durch Kombination dreier Achsen:

| Achse | Optionen |
|-------|----------|
| **Ausreißer** | Unbereinigt / IQR-Bereinigung / PCA-reduziert |
| **xG** | Ohne (25 Features) / Mit xG (34 Features) |
| **Feature-Selektion** | Alle Features / Hoch korrelierende entfernt |

**PCA-Analyse:** Visualisierung der erklärten Varianz, Scatterplot der ersten 2 Komponenten

#### Abschnitt 4: Model Training & Hyperparameter Tuning (Zelle 18-24)

**5 Modellklassen im Vergleich:**

| Modell | sklearn-Klasse | Zweck |
|--------|---------------|-------|
| Logistic Regression | `LogisticRegression` | Baseline (linear) |
| Random Forest | `RandomForestClassifier` | Ensemble-Baseline |
| LightGBM | `LGBMClassifier` | Gradient Boosting (leicht) |
| XGBoost | `XGBClassifier` | Hauptmodell (Gradient Boosting) |
| MLP | `MLPClassifier` | Neuronales Netz (nicht-linear) |

**Hyperparameter-Tuning:** `RandomizedSearchCV` mit `StratifiedKFold` (5 Folds)

**Train/Test-Split:**
- Training: Saison 2023-2024
- Test: Saison 2024-2025 / 2025-2026

**Klassen-Imbalance:** `compute_sample_weight("balanced")` gleicht H/D/A-Verteilung aus

**Ausgabe pro Variante:**
- Accuracy
- Classification Report (Precision/Recall/F1 pro Klasse)
- Konfusionsmatrix (als PNG gespeichert)

#### Abschnitt 5: Finale Auswertung (Zelle 25-30)

**Detaillierte Evaluation des besten Modells:**

1. Alle 5 Modelle auf Testset evaluieren
2. Metriken vergleichen: Accuracy, F1 macro, F1 draw, Log Loss
3. Beste Modelle speichern als `best_model_*.pkl`

**Gespeicherte Metadaten im Modell-Package:**
```python
{
    'model': trained_model,
    'scaler': fitted_scaler,
    'features': list_of_features,
    'le_classes': ['A', 'D', 'H'],  # LabelEncoder-Klassen
    'model_type': 'XGBClassifier',
    'variant': 'Mit xG, IQR-bereinigt',
    'metrics': {
        'accuracy': 0.567,
        'f1_macro': 0.543,
        'f1_draw': 0.312,
        'log_loss': 1.089
    },
    'saved_at': '2024-04-01 15:41:23'
}
```

---

## 📁 Ergebnisse

### Grafiken (`Ergebnisse/*.png`)

| Datei | Inhalt |
|-------|--------|
| `confusion_ohne_xG.png` | 2×2 Grid: LR + XGBoost Confusion Matrices (ohne xG) |
| `confusion_mit_xG.png` | 2×2 Grid: LR + XGBoost Confusion Matrices (mit xG) |
| `feature_importance_ohne_xG.png` | Top-15 Features (XGBoost, ohne xG) |
| `feature_importance_mit_xG.png` | Top-15 Features (XGBoost, mit xG) |
| `vergleich_accuracy_logloss.png` | Balken: Accuracy & Log-Loss aller Modelle |
| `vergleich_alle_confusion_matrices.png` | 4×3 Grid aller Confusion Matrices |
| `vergleich_radar.png` | Radar-Chart: Precision/Recall/F1 pro Klasse (H/D/A) |

### CSV-Exports (`Ergebnisse/*.csv`)

| Datei | Inhalt |
|-------|--------|
| `ergebnisse_ohne_xG.csv` | Accuracy, F1, Precision, Recall (ohne xG) |
| `ergebnisse_mit_xG.csv` | Accuracy, F1, Precision, Recall (mit xG) |
| `vergleich_gesamt.csv` | Alle 5 Modelle im Direktvergleich |

---

## 🔑 Technologie-Stack

| Bibliothek | Version (min) | Verwendung |
|-----------|---------------|------------|
| `pandas` | 1.3.0 | Datenladen, Merging, Feature Engineering, EWA-Berechnung |
| `numpy` | 1.21.0 | Numerische Berechnungen, Arrays, EWA-Formel |
| `scikit-learn` | 1.0.0 | LogisticRegression, RandomForest, StandardScaler, StratifiedKFold, RandomizedSearchCV, Metriken |
| `xgboost` | 1.5.0 | XGBClassifier (Hauptmodell: Gradient Boosting) |
| `lightgbm` | 3.3.0 | LGBMClassifier (Alternative zu XGBoost) |
| `matplotlib` | 3.4.0 | Alle Basis-Visualisierungen, Histogramme, Boxplots |
| `seaborn` | 0.11.0 | Heatmaps, Countplots, statistische Plots |
| `joblib` | 1.1.0 | Modell-Persistenz (Cache + Notebook-Export) |
| `difflib` | (stdlib) | Fuzzy Matching für Teamnamen-Eingaben |

---

## 📝 Nutzung im Alltag

### Schnellstart: Spielvorhersage

```bash
cd "Bachelor Arbeit"
python3 predict_spiel.py
# Eingabe: Heimteam (z.B. "Bayern")
# Eingabe: Auswärtsteam (z.B. "Dortmund")
```

### Notebook analysieren

```bash
cd "Bachelor Arbeit/xx_Notebook"
jupyter notebook notebook.ipynb
```

### Eigene Modelle trainieren

1. Notebook Zelle 1-22 ausführen (lädt Daten, trainiert alle 12 Varianten)
2. Zelle 28-30 ausführen (evaluiert + speichert beste Modelle)
3. Modelle liegen in `xx_Notebook/best_model_*.pkl`

---

## 📚 Weiterführende Dokumentation

- **`xx_Notebook/Features_Metriken.md`** – Detaillierte Erklärung aller 34 Features mit Formeln
- **`utils.py`** – Inline-Kommentare zu allen Funktionen
- **`predict_spiel.py`** – Docstrings zu allen Helper-Funktionen

---

## 📄 Lizenz & Zitierung

Dieses Projekt entstand im Rahmen einer Bachelorarbeit. Bei Nutzung oder Weiterverwendung bitte entsprechende Zitierung der Arbeit beachten.
