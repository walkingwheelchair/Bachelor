# Features & Metriken – Vollständige Dokumentation

Diese Datei dokumentiert **alle Features, Metriken und Evaluationskennzahlen** des Machine-Learning-Pipelines für die Bundesliga-Spielergebnisvorhersage.

---

## Inhaltsverzeichnis

1. [Zielvariable (FTR)](#1-zielvariable-ftr)
2. [EWA – Exponentially Weighted Average](#2-ewa---exponentially-weighted-average)
3. [BASE_FEATURES – 25 Basis-Features](#3-base_features---25-basis-features)
4. [XG_FEATURES – 9 zusätzliche xG-Features](#4-xg_features---9-zusätzliche-xg-features)
5. [Feature-Engineering Details](#5-feature-engineering-details)
6. [Evaluationsmetriken](#6-evaluationsmetriken)
7. [Data-Leakage-Prävention](#7-data-leakage-prävention)
8. [Die 12 Datensatz-Varianten](#8-die-12-datensatz-varianten)

---

## 1. Zielvariable (FTR)

**Full Time Result (FTR)** ist die Zielvariable des Klassifikationsproblems – das tatsächliche Spielergebnis.

### Kodierung

| Label | Numerisch | Bedeutung |
|-------|-----------|-----------|
| `H` (Home) | +1 | Heimsieg |
| `D` (Draw) | 0 | Unentschieden |
| `A` (Away) | −1 | Auswärtssieg |

### Klassenverteilung (typisch)

| Klasse | Absolute Häufigkeit | Relative Häufigkeit |
|--------|---------------------|---------------------|
| H (Heimsieg) | ~45% | Häufigste Klasse |
| A (Auswärtssieg) | ~30% | Zweithäufigste |
| D (Unentschieden) | ~25% | Seltenste Klasse |

**Herausforderung:** Klassen-Imbalance erfordert `class_weight="balanced"` oder `sample_weight` beim Training.

---

## 2. EWA – Exponentially Weighted Average

Das Herzstück des Feature-Engineerings. Der EWA berechnet einen gleitenden Durchschnitt über alle bisherigen Spiele einer Mannschaft, wobei **neuere Spiele stärker gewichtet** werden.

### Formel

```
EWA_neu = α × Wert_aktuell + (1 − α) × EWA_alt
```

mit:
```
α = 2 / (span + 1)
```

### Parameter

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `span` | 10 (Standard) | Fenstergröße für Gewichtung |
| `α` (alpha) | ~0.182 (bei span=10) | Gewichtungsfaktor für aktuellen Wert |

### Beispielrechnung

Ein Team schießt in den letzten 5 Spielen: 2, 1, 3, 0, 2 Tore

**Normaler Durchschnitt (letzte 5):** (2+1+3+0+2)/5 = 1.6

**EWA (rekursiv, span=10, α≈0.182):**
- Spiel 1: EWA = 2.0
- Spiel 2: EWA = 0.182×1 + 0.818×2 = 1.818
- Spiel 3: EWA = 0.182×3 + 0.818×1.818 = 2.034
- Spiel 4: EWA = 0.182×0 + 0.818×2.034 = 1.664
- Spiel 5: EWA = 0.182×2 + 0.818×1.664 = 1.725

**Vorteil gegenüber normalem Durchschnitt:**
- Alle Spiele fließen ein (kein "harter Cut" nach N Spielen)
- Neuere Spiele zählen mehr, ältere verblassen graduell
- Kein Informationsverlust durch Fensterbegrenzung

### Drei EWA-Kontexte

| Kontext | Datenquelle | Zweck |
|---------|-------------|-------|
| `overall` | Alle Spiele (Heim + Auswärts) | Gesamtform des Teams |
| `home_form` | Nur Heimspiele des Heimteams | Spezifische Heimstärke |
| `away_form` | Nur Auswärtsspiele des Auswärtsteams | Spezifische Auswärtsstärke |

### Pro Kontext berechnete Metriken (5 pro Kontext)

| Metrik | Beschreibung |
|--------|-------------|
| `ewa_GoalsScored` | Erwartete erzielte Tore (Angriffsstärke) |
| `ewa_GoalsConceded` | Erwartete kassierte Tore (Defensivschwäche) |
| `ewa_Points` | Erwartete Punkte (3=Sieg, 1=Unentschieden, 0=Niederlage) |
| `ewa_Shots` | Erwartete Schüsse insgesamt (Offensivaktivität) |
| `ewa_ShotsOnTarget` | Erwartete Schüsse aufs Tor (Qualität der Chancen) |

---

## 3. BASE_FEATURES – 25 Basis-Features

### Gruppe 1: Gesamtform (overall) – 10 Features

Diese Features beschreiben die allgemeine Form beider Teams über **alle Spiele** der laufenden Saison.

| Feature-ID | Beschreibung | Interpretation |
|------------|-------------|----------------|
| `home_team_overall_ewa_GoalsScored` | EWA der erzielten Tore (Heimteam, alle Spiele) | Wie viele Tore schießt das Heimteam im Durchschnitt? |
| `home_team_overall_ewa_GoalsConceded` | EWA der kassierten Tore (Heimteam, alle Spiele) | Wie viele Tore kassiert das Heimteam im Durchschnitt? |
| `home_team_overall_ewa_Points` | EWA der Punkte (Heimteam, alle Spiele) | Wie erfolgreich ist das Heimteam insgesamt? |
| `home_team_overall_ewa_Shots` | EWA der Schüsse (Heimteam, alle Spiele) | Wie aktiv ist das Heimteam offensiv? |
| `home_team_overall_ewa_ShotsOnTarget` | EWA der Schüsse aufs Tor (Heimteam, alle Spiele) | Wie präzise ist das Heimteam? |
| `away_team_overall_ewa_GoalsScored` | EWA der erzielten Tore (Auswärtsteam, alle Spiele) | Wie viele Tore schießt das Auswärtsteam? |
| `away_team_overall_ewa_GoalsConceded` | EWA der kassierten Tore (Auswärtsteam, alle Spiele) | Wie viele Tore kassiert das Auswärtsteam? |
| `away_team_overall_ewa_Points` | EWA der Punkte (Auswärtsteam, alle Spiele) | Wie erfolgreich ist das Auswärtsteam? |
| `away_team_overall_ewa_Shots` | EWA der Schüsse (Auswärtsteam, alle Spiele) | Wie aktiv ist das Auswärtsteam? |
| `away_team_overall_ewa_ShotsOnTarget` | EWA der Schüsse aufs Tor (Auswärtsteam, alle Spiele) | Wie präzise ist das Auswärtsteam? |

### Gruppe 2: Heimform (home_form) – 5 Features

Spezifisch für **Heimspiele des Heimteams**. Erfasst den Heimvorteil.

| Feature-ID | Beschreibung | Interpretation |
|------------|-------------|----------------|
| `home_team_home_form_ewa_GoalsScored` | EWA Tore (nur Heimspiele) | Wie stark ist das Heimteam zuhause im Angriff? |
| `home_team_home_form_ewa_GoalsConceded` | EWA kassierte Tore (nur Heimspiele) | Wie stabil ist die Heim-Defensive? |
| `home_team_home_form_ewa_Points` | EWA Punkte (nur Heimspiele) | Wie erfolgreich ist das Team zuhause? |
| `home_team_home_form_ewa_Shots` | EWA Schüsse (nur Heimspiele) | Wie aktiv ist das Team zuhause? |
| `home_team_home_form_ewa_ShotsOnTarget` | EWA Schüsse aufs Tor (nur Heimspiele) | Wie präzise ist das Team zuhause? |

### Gruppe 3: Auswärtsform (away_form) – 5 Features

Spezifisch für **Auswärtsspiele des Auswärtsteams**. Erfasst die Auswärtsstärke.

| Feature-ID | Beschreibung | Interpretation |
|------------|-------------|----------------|
| `away_team_away_form_ewa_GoalsScored` | EWA Tore (nur Auswärtsspiele) | Wie stark ist das Auswärtsteam im Angriff auswärts? |
| `away_team_away_form_ewa_GoalsConceded` | EWA kassierte Tore (nur Auswärtsspiele) | Wie stabil ist die Auswärts-Defensive? |
| `away_team_away_form_ewa_Points` | EWA Punkte (nur Auswärtsspiele) | Wie erfolgreich ist das Team auswärts? |
| `away_team_away_form_ewa_Shots` | EWA Schüsse (nur Auswärtsspiele) | Wie aktiv ist das Team auswärts? |
| `away_team_away_form_ewa_ShotsOnTarget` | EWA Schüsse aufs Tor (nur Auswärtsspiele) | Wie präzise ist das Team auswärts? |

### Gruppe 4: Differenz-Features – 2 Features

Direkter Vergleich der Teamstärken.

| Feature-ID | Formel | Interpretation |
|------------|--------|----------------|
| `goal_diff_avg` | `home_overall_GoalsScored − away_overall_GoalsScored` | Positiv = Heimteam trifft häufiger; Negativ = Auswärtsteam trifft häufiger |
| `points_diff_avg` | `home_overall_Points − away_overall_Points` | Positiv = Heimteam punktstärker; Negativ = Auswärtsteam punktstärker |

### Gruppe 5: Unentschieden-Features – 3 Features

Speziell zur besseren Vorhersage von Unentschieden (schwierigste Klasse).

| Feature-ID | Formel | Interpretation |
|------------|--------|----------------|
| `abs_goal_diff` | `│goal_diff_avg│` | Je kleiner (~0), desto ausgeglichener das Spiel → Unentschieden wahrscheinlicher |
| `abs_points_diff` | `│points_diff_avg│` | Je kleiner (~0), desto ausgeglichener die Punkte → Unentschieden wahrscheinlicher |
| `combined_draw_tendency` | `1/(1 + abs_goal_diff) × 1/(1 + abs_points_diff)` | Kombiniert beide Faktoren; Wert nahe 1.0 = beide Teams gleich stark = Unentschieden wahrscheinlich |

**Warum diese Features?** Unentschieden ist die seltenste Klasse (~25%) und am schwersten vorherzusagen. Diese Features helfen dem Modell, ausgeglichene Paarungen zu erkennen.

---

## 4. XG_FEATURES – 9 zusätzliche xG-Features

**Expected Goals (xG)** misst die Qualität von Torchancen unabhängig vom tatsächlichen Ergebnis.

### Saisonale xG-Daten (pro Team)

| Spalte | Beschreibung |
|--------|-------------|
| `xG_per_game` | Durchschnittlich erwartete eigene Tore pro Spiel (Angriffsstärke) |
| `xGA_per_game` | Durchschnittlich erwartete kassierte Tore pro Spiel (Defensivschwäche) |
| `xPTS_per_game` | Durchschnittlich erwartete Punkte pro Spiel (aus xG berechnet) |

### Abgeleitete xG-Features

| Feature-ID | Formel | Interpretation |
|------------|--------|----------------|
| `home_xG_per_game` | Direkt aus xG-Daten | Erwartete Tore des Heimteams pro Spiel |
| `home_xGA_per_game` | Direkt aus xG-Daten | Erwartete kassierte Tore des Heimteams pro Spiel |
| `home_xPTS_per_game` | Direkt aus xG-Daten | Erwartete Punkte des Heimteams pro Spiel |
| `away_xG_per_game` | Direkt aus xG-Daten | Erwartete Tore des Auswärtsteams pro Spiel |
| `away_xGA_per_game` | Direkt aus xG-Daten | Erwartete kassierte Tore des Auswärtsteams pro Spiel |
| `away_xPTS_per_game` | Direkt aus xG-Daten | Erwartete Punkte des Auswärtsteams pro Spiel |
| `xG_diff` | `home_xG_per_game − away_xG_per_game` | Positiv = Heimteam hat bessere Angriffs-xG |
| `xGA_diff` | `home_xGA_per_game − away_xGA_per_game` | Positiv = Heimteam hat schlechtere Defensive-xG |
| `abs_xG_diff` | `│xG_diff│` | Je kleiner, desto ausgeglichener → Unentschieden wahrscheinlicher |

### Was ist xG?

**Expected Goals** ist eine moderne Fußballstatistik, die jede Torchance basierend auf historischen Daten bewertet:

- **Faktoren:** Position, Winkel, Entfernung zum Tor, Körperteil, Vorlage-Typ
- **Wertebereich:** 0.0 (keine Chance) bis 1.0 (sichere Chance = Penalty ~0.76)
- **Interpretation:** xG = 1.5 bedeutet "aus diesen Chancen sollte man im Durchschnitt 1.5 Tore erzielen"

### Warum xG hilft

| Problem | Lösung durch xG |
|---------|-----------------|
| Glück/Pech bei der Verwertung | xG neutralisiert Zufallsschwankungen |
| Torwart-Überleistung | xG zeigt, ob Tore "verdient" waren |
| Kleine Stichprobe (wenige Spiele) | xG stabilisiert die Schätzung |

---

## 5. Feature-Engineering Details

### Team-Namen-Normalisierung

Verschiedene Datenquellen verwenden unterschiedliche Teamnamen. Das Mapping in `utils.py` normalisiert alle Varianten:

```python
TEAM_NAME_MAP = {
    "Bayern Munich": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RasenBallsport Leipzig",
    "M'gladbach": "Borussia M.Gladbach",
    # ... (vollständige Liste in utils.py)
}
```

### Punkte-Berechnung

```python
def calc_points(row):
    if row["FTR"] == "D": 
        return 1  # Unentschieden = 1 Punkt
    if row["is_home"] == 1 and row["FTR"] == "H": 
        return 3  # Heimsieg = 3 Punkte
    if row["is_home"] == 0 and row["FTR"] == "A": 
        return 3  # Auswärtssieg = 3 Punkte
    return 0  # Niederlage = 0 Punkte
```

### EWA-Berechnung (Implementierung)

```python
def compute_ewa(data, span):
    grouped = data.groupby("Team")
    shifted = grouped.shift(1)  # Vergangene Spiele (kein Look-Ahead!)
    ewa = shifted[["GoalsScored", "GoalsConceded", "Points", 
                   "Shots", "ShotsOnTarget"]].groupby(data["Team"]).ewm(
        span=span, min_periods=1).mean()
    return ewa.reset_index(level=0, drop=True)
```

**Wichtig:** `shift(1)` verhindert Data Leakage – nur vergangene Spiele fließen ein.

---

## 6. Evaluationsmetriken

Da es sich um ein **3-Klassen-Klassifikationsproblem** (H/D/A) handelt, werden alle Metriken pro Klasse sowie aggregiert berechnet.

### 6.1 Konfusionsmatrix (Confusion Matrix)

Die Basis aller Metriken. Zeigt für jede tatsächliche Klasse, wie das Modell sie vorhergesagt hat.

```
                    Vorhergesagt
                    H       D       A
                ┌───────┬───────┬───────┐
Tatsächlich H   │ TP_H  │ FN_HD │ FN_HA │
                ├───────┼───────┼───────┤
Tatsächlich D   │ FP_DH │ TP_D  │ FN_DA │
                ├───────┼───────┼───────┤
Tatsächlich A   │ FP_AH │ FP_AD │ TP_A  │
                └───────┴───────┴───────┘
```

**Begriffe (am Beispiel Klasse H = Heimsieg):**

| Begriff | Bedeutung | Beispiel |
|---------|-----------|----------|
| TP (True Positive) | Modell sagt H, tatsächlich H ✅ | 25 korrekte Heimsieg-Vorhersagen |
| FP (False Positive) | Modell sagt H, tatsächlich D oder A ❌ | 10 Fehlalarme (sagt H, war D/A) |
| FN (False Negative) | Modell sagt nicht H, tatsächlich H ❌ | 15 übersehene Heimsiege |
| TN (True Negative) | Modell sagt nicht H, tatsächlich kein H ✅ | Alle korrekten Nicht-H-Vorhersagen |

### 6.2 Accuracy (Trefferquote)

**Frage:** Wie viel Prozent aller Spiele wurden korrekt vorhergesagt?

```
             TP_H + TP_D + TP_A
Accuracy = ────────────────────────────
                N_gesamt
```

**Beispiel:** 100 Spiele, 55 korrekt → Accuracy = 55%

**Limitation:** Bei Klassen-Imbalance kann Accuracy irreführend sein. Ein Modell, das immer "H" vorhersagt, erreicht ~45% Accuracy (Baseline).

### 6.3 Precision (Genauigkeit)

**Frage:** Wenn das Modell "H" vorhersagt – wie oft stimmt das wirklich?

```
              TP
Precision = ────────
            TP + FP
```

**Beispiel für Klasse H:**
- Modell sagt 40× "H" vorher
- Davon tatsächlich Heimsiege: 28
- **Precision(H) = 28/40 = 0.70 (70%)**

**Interpretation:** 70% der "Heimsieg"-Vorhersagen waren korrekt. 30% waren Fehlalarme.

### 6.4 Recall (Trefferrate / Sensitivität)

**Frage:** Von allen tatsächlichen Heimsiegen – wie viele hat das Modell gefunden?

```
             TP
Recall = ──────────
          TP + FN
```

**Beispiel für Klasse H:**
- 50 echte Heimsiege im Testset
- Modell hat 28 davon erkannt
- **Recall(H) = 28/50 = 0.56 (56%)**

**Interpretation:** Das Modell erkennt 56% aller Heimsiege. 44% werden übersehen (als D oder A vorhergesagt).

### 6.5 Precision vs. Recall – Der Zielkonflikt

| Modellverhalten | Precision | Recall |
|----------------|-----------|--------|
| Vorsichtig (sagt selten "H") | Hoch (wenige Fehlalarme) | Niedrig (viele übersehen) |
| Optimistisch (sagt oft "H") | Niedrig (viele Fehlalarme) | Hoch (wenige übersehen) |

### 6.6 F1-Score (Harmonisches Mittel)

**Frage:** Wie balanciert sind Precision und Recall?

```
          2 × Precision × Recall
F1 = ────────────────────────────────────
       Precision + Recall


     2 × TP
= ───────────────────
  2×TP + FP + FN
```

**Beispiel:**
| Precision | Recall | F1 |
|-----------|--------|-----|
| 1.00 | 0.00 | 0.00 |
| 0.70 | 0.70 | 0.70 |
| 0.90 | 0.50 | 0.64 |
| 0.70 | 0.56 | 0.62 |

**Interpretation:** F1 bestraft extreme Ungleichgewichte. Ein F1 von 0.0 entsteht, wenn Precision ODER Recall 0 ist.

### 6.7 Aggregation über Klassen

Da wir 3 Klassen haben (H/D/A), gibt es zwei Aggregationsmethoden:

#### Macro-Average

Einfacher Durchschnitt – alle Klassen gleich gewichtet:

```
Macro-F1 = (F1_H + F1_D + F1_A) / 3
```

**Wann wichtig:** Wenn jede Klasse gleich wichtig ist (auch die seltene Klasse D).

#### Weighted-Average

Gewichtet nach Klassenhäufigkeit:

```
                F1_H × n_H + F1_D × n_D + F1_A × n_A
Weighted-F1 = ──────────────────────────────────────────
                            N_gesamt
```

**Wann wichtig:** Für die reale Gesamtperformance (häufige Klassen zählen mehr).

### 6.8 Log Loss (Logarithmischer Verlust)

**Frage:** Wie gut sind die vorhergesagten Wahrscheinlichkeiten kalibriert?

```
Log Loss = −1/N × Σ [y_true × log(y_pred)]
```

Für 3 Klassen (Multi-Class):

```
Log Loss = −1/N × Σ Σ [y_c × log(p_c)]
                    c∈{H,D,A}
```

**Interpretation:**
- Perfekte Vorhersage: Log Loss = 0
- Zufallsrate (33% pro Klasse): Log Loss ≈ 1.1
- Typische Werte: 0.8 – 1.2

**Vorteil gegenüber Accuracy:** Log Loss bestraft falsche Vorhersagen mit hoher Konfidenz stärker.

### 6.9 Übersicht aller Metriken

| Metrik | Frage | Gut wenn... | Typischer Bereich |
|--------|-------|-------------|-------------------|
| **Accuracy** | Wie oft richtig insgesamt? | > 50% (Baseline ~45%) | 0.45 – 0.65 |
| **Precision** | Wenn ich H tippe, stimmt das? | Hoch (> 0.6) | 0.4 – 0.8 |
| **Recall** | Finde ich alle H? | Hoch (> 0.5) | 0.3 – 0.7 |
| **F1-Score** | Balance aus beidem? | Hoch (> 0.5) | 0.3 – 0.7 |
| **Macro-F1** | Gut für alle 3 Klassen? | Hoch (> 0.4) | 0.3 – 0.6 |
| **Weighted-F1** | Gesamtperformance? | Hoch (> 0.5) | 0.4 – 0.6 |
| **Log Loss** | Gute Wahrscheinlichkeiten? | Niedrig (< 1.1) | 0.8 – 1.3 |

---

## 7. Data-Leakage-Prävention

### Problem

xG-Saisonwerte stehen erst am Saisonende fest. Würden wir sie für Spiele während der Saison verwenden, wäre das **Data Leakage** (Information aus der Zukunft).

### Lösung: Lag-1-Ansatz

| Saison-Typ | xG-Quelle | Begründung |
|------------|-----------|------------|
| Abgeschlossene Saisons (z.B. 2022/23) | xG der Vorsaison (2021/22) | Simuliert den Wissensstand während der Saison |
| Aktuelle Saison (z.B. 2025/26) | Direkte xG-Werte | Diese werden wöchentlich aktualisiert und sind "live" verfügbar |

### Implementierung (utils.py)

```python
def merge_xg_features(match_df, xg_df):
    all_seasons = sorted(xg_df["Season"].unique())
    max_season = all_seasons[-1]  # Aktuellste Saison
    
    for season in all_seasons:
        if season == max_season:
            # Aktuelle Saison: direkte Nutzung
            prior = season
        else:
            # Historische Saison: Vorjahreswert als Prior
            prior = all_seasons[all_seasons.index(season) - 1]
        
        # xG-Daten der Prior-Saison für diese Saison verwenden
        ...
```

### EWA: Shift(1) verhindert Look-Ahead

```python
shifted = grouped.shift(1)  # Nur vergangene Spiele!
```

Ohne `shift(1)` würde das aktuelle Spiel in seinen eigenen EWA-Wert einfließen – ebenfalls Data Leakage.

---

## 8. Die 12 Datensatz-Varianten

Das Notebook vergleicht systematisch 12 Kombinationen:

### Achsen der Variation

| Achse | Optionen | Beschreibung |
|-------|----------|--------------|
| **Algorithmus** | LogReg, RF, LGBM, XGBoost, MLP | 5 verschiedene Modellklassen |
| **Ausreißer** | Unbereinigt / IQR / PCA | Umgang mit Extremwerten |
| **xG** | Ohne (25 Features) / Mit (34 Features) | Expected Goals einbezogen? |
| **Feature-Selektion** | Alle / Korrelation-bereinigt | Redundante Features entfernen |

### IQR-Ausreißer-Bereinigung

Entfernt Spiele mit extremen Werten (z.B. 8:0 Ergebnisse):

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Behalte nur Zeilen innerhalb von 1.5 × IQR
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

### PCA-Reduktion

Reduziert Features auf Hauptkomponenten mit 95% erklärter Varianz:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
```

### Alle 12 Varianten im Überblick

| ID | Algorithmus | Ausreißer | xG | Feature-Selektion | Features |
|----|-------------|-----------|-----|-------------------|----------|
| 1 | LogReg | Unbereinigt | Nein | Alle | 25 |
| 2 | LogReg | Unbereinigt | Ja | Alle | 34 |
| 3 | LogReg | IQR | Nein | Alle | 25 |
| 4 | LogReg | IQR | Ja | Alle | 34 |
| 5 | RF | Unbereinigt | Nein | Alle | 25 |
| 6 | RF | Unbereinigt | Ja | Alle | 34 |
| 7 | RF | IQR | Nein | Alle | 25 |
| 8 | RF | IQR | Ja | Alle | 34 |
| 9 | LGBM | Unbereinigt | Nein | Alle | 25 |
| 10 | LGBM | Unbereinigt | Ja | Alle | 34 |
| 11 | XGBoost | Unbereinigt | Nein | Alle | 25 |
| 12 | XGBoost | Unbereinigt | Ja | Alle | 34 |

*(Hinweis: Die tatsächliche Anzahl kann je nach Notebook-Konfiguration variieren)*

---

## Anhang: Feature-Listen (Copy-Paste für Code)

### BASE_FEATURES (25 Features)

```python
BASE_FEATURES = [
    # Overall (10 Features)
    "home_team_overall_ewa_GoalsScored",
    "home_team_overall_ewa_GoalsConceded",
    "home_team_overall_ewa_Points",
    "home_team_overall_ewa_Shots",
    "home_team_overall_ewa_ShotsOnTarget",
    "away_team_overall_ewa_GoalsScored",
    "away_team_overall_ewa_GoalsConceded",
    "away_team_overall_ewa_Points",
    "away_team_overall_ewa_Shots",
    "away_team_overall_ewa_ShotsOnTarget",
    
    # Home Form (5 Features)
    "home_team_home_form_ewa_GoalsScored",
    "home_team_home_form_ewa_GoalsConceded",
    "home_team_home_form_ewa_Points",
    "home_team_home_form_ewa_Shots",
    "home_team_home_form_ewa_ShotsOnTarget",
    
    # Away Form (5 Features)
    "away_team_away_form_ewa_GoalsScored",
    "away_team_away_form_ewa_GoalsConceded",
    "away_team_away_form_ewa_Points",
    "away_team_away_form_ewa_Shots",
    "away_team_away_form_ewa_ShotsOnTarget",
    
    # Difference Features (2 Features)
    "goal_diff_avg",
    "points_diff_avg",
    
    # Draw Tendency Features (3 Features)
    "abs_goal_diff",
    "abs_points_diff",
    "combined_draw_tendency",
]
```

### XG_FEATURES (34 Features = BASE + 9 xG)

```python
XG_FEATURES = BASE_FEATURES + [
    # xG Features (6 direkte Werte)
    "home_xG_per_game",
    "home_xGA_per_game",
    "home_xPTS_per_game",
    "away_xG_per_game",
    "away_xGA_per_game",
    "away_xPTS_per_game",
    
    # xG Difference Features (3 Features)
    "xG_diff",
    "xGA_diff",
    "abs_xG_diff",
]
```

---

## Quellen & Weiterführendes

- **EWA (Exponential Smoothing):** https://en.wikipedia.org/wiki/Exponential_smoothing
- **xG (Expected Goals):** https://understat.com/ (Datenquelle)
- **scikit-learn Metriken:** https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- **XGBoost:** https://xgboost.readthedocs.io/

---

*Diese Dokumentation dient als Referenz für KI-Assistenten und menschliche Entwickler, um alle Features und Metriken des Bundesliga-Vorhersagesystems vollständig zu verstehen.*
