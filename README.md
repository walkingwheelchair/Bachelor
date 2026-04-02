# ⚽ Vorhersage von Bundesliga-Spielergebnissen – Bachelorarbeit

Dieses Repository enthält den vollständigen Code einer wissenschaftlichen Bachelorarbeit, die untersucht, ob und wie stark die Integration von **Expected Goals (xG)** die Vorhersagegenauigkeit von Bundesliga-Spielergebnissen verbessert.

---

## 📋 Inhaltsverzeichnis

1. [Wissenschaftliche Fragestellung](#-wissenschaftliche-fragestellung)
2. [Projektstruktur](#-projektstruktur)
3. [Datenbasis](#-datenbasis)
4. [utils.py – Datenpipeline](#-utilspy--datenpipeline)
5. [notebook.ipynb – ML-Pipeline](#-notebookipynb--ml-pipeline)
6. [predict_spiel.py – Live-Vorhersage](#-predict_spielpy--live-vorhersage)
7. [Ergebnisse](#-ergebnisse-ordner)
8. [Installation & Ausführung](#-installation--ausführung)

---

## 🔬 Wissenschaftliche Fragestellung

**Kernfrage:** Verbessert die Integration von Expected-Goals-Daten (xG) die Vorhersagegenauigkeit eines Machine-Learning-Modells für Bundesliga-Spielergebnisse signifikant?

Der **Full Time Result (FTR)** ist die Zielvariable:
- `H` = Heimsieg
- `D` = Unentschieden
- `A` = Auswärtssieg

Das Projekt vergleicht **8 Datensatz-Varianten** (mit/ohne xG, mit/ohne Ausreißer-Bereinigung, mit/ohne Feature-Selektion) und **2 Modellklassen** (Logistic Regression, XGBoost).

---

## 📂 Projektstruktur

```
Bachelor Arbeit/
│
├── utils.py                       # Zentrale Datenpipeline & Feature Engineering
├── predict_spiel.py               # Interaktives CLI-Tool für Live-Vorhersagen
│
├── Bundesliga_Quellen/            # Rohdaten: Spielergebnisse 2016–2026 (10 CSVs)
│   ├── Bundesliga 2016-2017.csv
│   ├── ...
│   └── Bundesliga 2025-2026.csv
│
├── xG_Quellen/                    # xG-Saisondaten pro Team 2016–2026 (10 CSVs)
│   ├── xG 2016-2017.csv
│   ├── ...
│   └── xG 2025-2026.csv
│
├── xx_Notebook/
│   ├── notebook.ipynb             # Vollständige ML-Pipeline (EDA → Training → Evaluation)
│   └── Kennzahlen_Erklaerung.md   # Dokumentation aller Features & Metriken (mit Formeln)
│
├── Ergebnisse/                    # Automatisch generierte Grafiken & Auswertungen
│   ├── confusion_mit_xG.png / confusion_ohne_xG.png
│   ├── feature_importance_mit_xG.png / feature_importance_ohne_xG.png
│   ├── vergleich_accuracy_logloss.png
│   ├── vergleich_alle_confusion_matrices.png
│   ├── vergleich_radar.png
│   ├── ergebnisse_mit_xG.csv / ergebnisse_ohne_xG.csv
│   └── vergleich_gesamt.csv
│
└── model_cache/                   # Gecachte Modelle (automatisch erstellt)
    ├── models_ohne_xG.joblib / models_mit_xG.joblib
    ├── label_encoder.joblib / features.joblib
    └── meta.json
```

---

## 📊 Datenbasis

### `Bundesliga_Quellen/` – Spielergebnisse (10 CSV-Dateien, Saisons 2016/17–2025/26)

Jede CSV enthält alle Bundesliga-Spiele einer Saison (~306 Spiele, 18 Teams). Genutzte Spalten:

| Spalte | Beschreibung |
|--------|-------------|
| `Date` | Spieltag |
| `HomeTeam` / `AwayTeam` | Teamnamen |
| `FTHG` / `FTAG` | Tore Heim/Auswärts (Full Time Home/Away Goals) |
| `FTR` | Ergebnis: `H`, `D` oder `A` – die **Zielvariable** |
| `HS` / `AS` | Abgegebene Schüsse Heim/Auswärts |
| `HST` / `AST` | Schüsse aufs Tor Heim/Auswärts |

**Umfang:** 10 Saisons, ~2.997 Spiele gesamt.

---

### `xG_Quellen/` – Expected Goals Statistiken (10 CSV-Dateien)

Jede CSV enthält saisonbasierte xG-Werte **pro Team** (18 Einträge pro Datei):

| Spalte | Beschreibung |
|--------|-------------|
| `Team` | Teamname |
| `Season` | Saison (z.B. `2023-2024`) |
| `xG_per_game` | Ø erwartete eigene Tore pro Spiel (Angriffsstärke) |
| `xGA_per_game` | Ø erwartete kassierte Tore pro Spiel (Defensivschwäche) |
| `xPTS_per_game` | Ø erwartete Punkte pro Spiel (aus xG berechnet) |

**xG (Expected Goals)** misst, wie viele Tore eine Mannschaft basierend auf der Schussqualität *eigentlich* hätte erzielen sollen – unabhängig von Glück oder Torwartleistungen.

**Data-Leakage-Prävention:** Da Saisonwerte erst am Saisonende feststehen, wird für abgeschlossene Saisons der xG-Wert der **Vorsaison (Lag-1)** als Prior genutzt. Nur für die aktuell laufende Saison werden die wöchentlich aktualisierten Daten direkt verwendet.

---

## ⚙️ `utils.py` – Datenpipeline

Die zentrale Bibliothek des Projekts. Wird von `notebook.ipynb` und `predict_spiel.py` importiert.

### Konstanten

```python
BUNDESLIGA_DIR  # Absoluter Pfad zu Bundesliga_Quellen/
XG_DIR          # Absoluter Pfad zu xG_Quellen/
BASE_FEATURES   # Liste der 25 Basis-Features (ohne xG)
XG_FEATURES     # Liste der 34 Features inkl. xG-Erweiterungen
```

### Funktionen

#### `load_bundesliga_data() → pd.DataFrame`
Lädt alle 10 CSV-Dateien aus `Bundesliga_Quellen/`, ergänzt eine `Season`-Spalte und normalisiert Teamnamen (einheitliche Schreibweise). Gibt einen kombinierten DataFrame aller Saisons zurück (~2.997 Zeilen).

#### `load_xg_data() → pd.DataFrame`
Lädt alle 10 CSV-Dateien aus `xG_Quellen/`, normalisiert Teamnamen und gibt einen kombinierten DataFrame zurück (180 Zeilen: 18 Teams × 10 Saisons).

#### `normalize_team(name: str) → str`
Normalisiert Teamnamen für konsistente Joins zwischen den Datenquellen (z.B. verschiedene Schreibweisen wie „Bayer 04" vs. „Bayer Leverkusen").

#### `compute_rolling_features(df) → pd.DataFrame`
**Das Feature-Engineering-Herzstück.** Berechnet für jedes Spiel die historische Form beider Teams mittels **EWA (Exponentially Weighted Average)**. Drei Kontexte werden getrennt berechnet:

| Kontext | Beschreibung |
|---------|-------------|
| `overall` | EWA über alle Spiele der Saison (Heim + Auswärts) |
| `home_form` | EWA nur über bisherige **Heimspiele** des Heimteams |
| `away_form` | EWA nur über bisherige **Auswärtsspiele** des Auswärtsteams |

Je Kontext werden folgende Kennzahlen als EWA berechnet:
`GoalsScored`, `GoalsConceded`, `Points`, `Shots`, `ShotsOnTarget`

Das ergibt **10 Features pro Team × 2 Teams = 20 EWA-Features** pro Spiel.

> **EWA-Formel:** `EWA_neu = α × Wert_aktuell + (1 − α) × EWA_alt`  
> mit `α = 2 / (span + 1)`. Neuere Spiele zählen stärker; kein Spiel fällt komplett weg.

Die getrennten Heim/Auswärts-Kontexte ermöglichen es dem Modell, den **Heimvorteil** zu erkennen.

#### `merge_xg_features(match_df, xg_df) → pd.DataFrame`
Fügt xG-Saisondaten per Join (Team + Saison) als Features hinzu. Implementiert die **Lag-1-Logik**: Für abgeschlossene Saisons wird der Vorjahreswert als Prior verwendet, für die aktuelle Saison der aktuelle Wert.

#### `add_derived_features(df, include_xg=False) → pd.DataFrame`
Berechnet abgeleitete Differenz- und Unentschieden-Features:

| Feature | Formel | Bedeutung |
|---------|--------|-----------|
| `goal_diff_avg` | `home_EWA_Goals − away_EWA_Goals` | Positiv = Heimteam trifft häufiger |
| `points_diff_avg` | `home_EWA_Points − away_EWA_Points` | Positiv = Heimteam punktstärker |
| `abs_goal_diff` | `│goal_diff_avg│` | Je kleiner → ausgeglichener |
| `abs_points_diff` | `│points_diff_avg│` | Je kleiner → ausgeglichener |
| `combined_draw_tendency` | `1/(1+abs_goal) × 1/(1+abs_pts)` | Nahe 1.0 → Unentschieden wahrscheinlicher |
| `xG_diff` | `home_xG − away_xG` | *(nur mit xG)* Angriffsunterschied |
| `xGA_diff` | `home_xGA − away_xGA` | *(nur mit xG)* Defensivunterschied |
| `abs_xG_diff` | `│xG_diff│` | *(nur mit xG)* Ausgeglichenheit xG |

Die letzten 3 Unentschieden-Features (`abs_goal_diff`, `abs_points_diff`, `combined_draw_tendency`) wurden speziell hinzugefügt, da Unentschieden die schwierigste Klasse zur Vorhersage ist.

#### `prepare_dataset(include_xg=False) → (DataFrame, list)`
Komplette Pipeline in einer Funktion: Daten laden → EWA berechnen → xG mergen (optional) → abgeleitete Features → NaN-Zeilen entfernen. Gibt den fertigen DataFrame + Feature-Liste zurück.

---

## 📓 `xx_Notebook/notebook.ipynb` – ML-Pipeline

Das Jupyter Notebook ist das **wissenschaftliche Kernstück** und führt die vollständige Analyse in 4 Abschnitten durch.

### Abschnitt 1: Daten laden & vorbereiten
Nutzt `utils.py`, um alle Spieldaten zu laden, EWA-Features zu berechnen und xG-Daten zu joinen. Erzeugt zwei Basis-DataFrames: `df_base` (ohne xG, ~2.955 Spiele) und `df_with_xg` (mit xG, ~2.280 Spiele).

### Abschnitt 2: Data Exploration (EDA)

| Visualisierung | Inhalt |
|---------------|--------|
| Balkendiagramm FTR | Verteilung von H / D / A (zeigt Heimvorteil ~45%) |
| Histogramme (25×) | Verteilung jeder einzelnen Variable |
| Boxplots (25×) | Ausreißer-Erkennung pro Variable |
| Scatter Plots (25×) | Beziehung jeder Variable mit dem Target FTR |
| Korrelationsmatrix | Heatmap aller Feature-Korrelationen inkl. Zielvariable |

### Abschnitt 3: Datensatz-Varianten (8 DataFrames)

Systematische Erstellung von 8 Varianten durch Kombination dreier Achsen:

| Achse | Option A | Option B |
|-------|---------|---------|
| Ausreißer | Unbereinigt | IQR-Bereinigung |
| xG | Ohne (25 Features) | Mit xG (34 Features) |
| Feature-Selektion | Alle Features | Hoch korrelierende Features |

→ `df1` bis `df8`, jeder mit passender Feature-Liste.

### Abschnitt 4: Modellierung & Evaluation

- **Modell:** XGBoost Klassifikator (`XGBClassifier`)
- **Tuning:** `RandomizedSearchCV` mit `StratifiedKFold` (5 Folds) – automatische Hyperparameter-Optimierung
- **Train/Test:** Saison 2023-2024 als Trainingsset; Saison 2024-2025 / 2025-2026 als Testset
- **Klassen-Imbalance:** `compute_sample_weight("balanced")` gleicht die ungleiche Verteilung von H/D/A aus
- **Ausgabe pro Variante:** Accuracy, Classification Report (Precision/Recall/F1 pro Klasse), Konfusionsmatrix

Die Ausgaben (Grafiken + CSV) werden im Ordner `Ergebnisse/` gespeichert.

> Eine vollständige Erklärung aller Features und Metriken (inkl. mathematischer Formeln) findet sich in `xx_Notebook/Kennzahlen_Erklaerung.md`.

---

## 🖥️ `predict_spiel.py` – Live-Vorhersage

Ein **interaktives Kommandozeilen-Tool** für Vorhersagen zukünftiger Bundesliga-Spiele. Es trainiert 4 Modelle auf allen verfügbaren historischen Daten und gibt für eine eingegebene Paarung eine strukturierte Prognose aller 4 Modelle + Konsens aus.

### Die 4 Modelle

| Modell | Algorithmus | Feature-Set |
|--------|------------|------------|
| 1 | Logistic Regression | 25 BASE_FEATURES (ohne xG) |
| 2 | XGBoost | 25 BASE_FEATURES (ohne xG) |
| 3 | Logistic Regression | 34 XG_FEATURES (mit xG) |
| 4 | XGBoost | 34 XG_FEATURES (mit xG) |

Alle Modelle nutzen Klassen-Gewichtung (`balanced`) um die ungleiche Verteilung von H/D/A auszugleichen. XGBoost nutzt `compute_sample_weight`, Logistic Regression `class_weight="balanced"`.

### Modell-Cache (`model_cache/`)

Modelle werden nach dem Training mit `joblib` auf der Festplatte gespeichert. Beim nächsten Start prüft das Tool via **mtime-Vergleich der CSV-Dateien**, ob Re-Training nötig ist. Falls keine CSV-Datei neuer ist als der Cache → sofortiger Start in Sekunden.

| Datei | Inhalt |
|-------|--------|
| `models_ohne_xG.joblib` | LR + XGBoost ohne xG |
| `models_mit_xG.joblib` | LR + XGBoost mit xG |
| `label_encoder.joblib` | LabelEncoder H/D/A ↔ 0/1/2 |
| `features.joblib` | Feature-Listen beider Varianten |
| `meta.json` | Timestamp der CSV-Dateien beim letzten Training |

### Team-Suche (Fuzzy Matching)
Eingaben wie `"Bayern"` oder `"Dortmnd"` werden via `difflib.get_close_matches` automatisch auf den korrekten Teamnamen gemapped.

### Aktuell-Form für zukünftige Spiele
Der EWA für das bevorstehende Spiel wird berechnet, indem die Paarung als Dummy-Zeile ans Ende des DataFrames angehängt wird – so erhält man die aktuelle gewichtete Form beider Teams ohne tatsächliche Spielergebnisse zu kennen.

### Ausgabe-Beispiel

```
=================================================================
  BUNDESLIGA LIVE-VORHERSAGE (MIT XGBOOST & EWA)
=================================================================

📊 FORM (Exponentially Weighted Averages):
                               HEIM      AUSWÄRTS
   Gew. Ø Tore erzielt         2.31          1.87
   Gew. Ø Tore kassiert        0.94          1.23
   Gew. Ø Punkte/Spiel         2.41          1.98

📈 xG-SAISONDATEN (Vorjahreswerte als Prior – kein Leakage):
   xG / Spiel                  2.18          1.92

─────────────────────────────────────────────────────────────────
  Modell                         Tip      P(H)    P(D)    P(A)
  Logistic Regression [OHNE xG]  🏠 Heimsieg   52.1%   24.3%   23.6%
  XGBoost [OHNE xG]              🏠 Heimsieg   58.4%   21.1%   20.5%
  Logistic Regression [MIT xG]   🏠 Heimsieg   54.7%   22.8%   22.5%
  XGBoost [MIT xG]               🏠 Heimsieg   61.2%   19.4%   19.4%
─────────────────────────────────────────────────────────────────
  🗳️  KONSENS (4/4 Modelle): 🏠 Heimsieg
─────────────────────────────────────────────────────────────────
```

### Kommandozeilen-Argumente

```bash
# Interaktiver Modus (Standard)
python3 predict_spiel.py

# Direkte Eingabe ohne Interaktion
python3 predict_spiel.py --heim Bayern --auswaerts Dortmund

# Cache ignorieren und Modelle neu trainieren
python3 predict_spiel.py --no-cache

# Bestimmte Saison erzwingen
python3 predict_spiel.py --saison 2024-2025
```

---

## 📁 `Ergebnisse/` Ordner

Alle Ausgaben der Notebook-Pipeline werden hier gespeichert:

| Datei | Inhalt |
|-------|--------|
| `confusion_ohne_xG.png` | Konfusionsmatrizen (LR + XGBoost) für Modelle ohne xG |
| `confusion_mit_xG.png` | Konfusionsmatrizen für Modelle mit xG |
| `feature_importance_ohne_xG.png` | Top-Feature-Importances (XGBoost, ohne xG) |
| `feature_importance_mit_xG.png` | Top-Feature-Importances (XGBoost, mit xG) |
| `vergleich_accuracy_logloss.png` | Balkendiagramm: Accuracy & Log-Loss aller 4 Modelle |
| `vergleich_alle_confusion_matrices.png` | 2×2-Grid aller Konfusionsmatrizen im Vergleich |
| `vergleich_radar.png` | Radar-Chart: Precision/Recall pro Klasse (H/D/A) für alle 4 Modelle |
| `ergebnisse_ohne_xG.csv` | Numerische Metriken (Accuracy, F1 etc.) ohne xG |
| `ergebnisse_mit_xG.csv` | Numerische Metriken mit xG |
| `vergleich_gesamt.csv` | Alle 4 Modelle im direkten Vergleich |

---

## 🚀 Installation & Ausführung

### Voraussetzungen

Python 3.9+ sowie:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

### Jupyter Notebook starten (EDA & ML-Pipeline)

```bash
cd "Bachelor Arbeit/xx_Notebook"
jupyter notebook notebook.ipynb
```

### Interaktives Vorhersage-Tool starten

```bash
cd "Bachelor Arbeit"
python3 predict_spiel.py
```

Beim ersten Start: ~1–2 Minuten Training aller 4 Modelle. Danach gecacht → nächster Start in Sekunden.

---

## 🔑 Technologie-Stack

| Bibliothek | Verwendung |
|-----------|-----------|
| `pandas` | Datenladen, Merging, Feature Engineering |
| `numpy` | Numerische Berechnungen, EWA |
| `scikit-learn` | LogisticRegression, StandardScaler, StratifiedKFold, RandomizedSearchCV, Metriken |
| `xgboost` | XGBClassifier (Hauptmodell Gradient Boosting) |
| `matplotlib` / `seaborn` | EDA-Visualisierungen, Konfusionsmatrizen, Radar-Charts |
| `joblib` | Modell-Persistenz (Cache) |
| `difflib` | Fuzzy Matching für Teamnamen-Eingaben |