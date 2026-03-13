# Bundesliga Match Prediction – Bachelor Thesis

Dieses Repository enthält den Code für eine Bachelorarbeit zur Vorhersage von Bundesliga-Spielergebnissen. Der Fokus liegt auf dem Vergleich traditioneller Leistungsdaten mit **Expected Goals (xG)** als Feature, um deren Einfluss auf die Vorhersagegenauigkeit zu analysieren.

## 📊 Projektübersicht

Das Ziel ist die Vorhersage des Spielausgangs (**Heimsieg, Unentschieden, Auswärtssieg**). 
Dazu werden vier Modell-Varianten verglichen:
1. **Logistic Regression (ohne xG)**
2. **Random Forest (ohne xG)**
3. **Logistic Regression (mit xG)**
4. **Random Forest (mit xG)**

### Kernergebnisse
In den Tests (Saisons 2023/24 – 2025/26) konnte gezeigt werden, dass die Integration von xG-Daten die Genauigkeit der Modelle um **ca. 4-5% verbessert**.

---

## 📂 Struktur & Daten

- `Bundesliga_Quellen/`: Historische Spieldaten (Tore, Schüsse, etc.) seit 2016/17.
- `xG_Quellen/`: Saisonbasierte Expected-Goals-Daten pro Team.
- `Ergebnisse/`: Automatisch generierte Visualisierungen (Plots) und CSV-Zusammenfassungen.
- `utils.py`: Zentrale Pipeline für Datenladen, Team-Mapping und Rolling-Window Feature Engineering (letzte 20 Spiele).

---

## 🛠 Installation

Stelle sicher, dass Python 3.9+ installiert ist. Installiere die benötigten Bibliotheken:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 🚀 Ausführung

Es gibt drei Haupt-Einstiegspunkte:

### 1. Der vollständige Vergleich (`vergleich.py`)
Dies ist das Hauptskript für die Analyse. Es trainiert alle Modelle, evaluiert sie auf den Test-Saisons und erstellt alle Vergleichs-Visualisierungen.

```bash
python3 vergleich.py
```
*Erzeugte Plots in `Ergebnisse/`: Accuracy-Vergleich, Confusion Matrices, Radar-Chart und Feature Importance.*

### 2. Das interaktive Live-Tool (`predict_spiel.py`)
Möchtest du ein Spiel von "heute Abend" vorhersagen? Dieses Tool trainiert die Modelle auf allen verfügbaren Daten und lässt dich Teams frei eingeben.

```bash
python3 predict_spiel.py
```
- **Tipp:** Du kannst Kurznamen wie "Bayern", "BVB" oder "Gladbach" verwenden.
- Das Tool zeigt dir die Form der Teams, erwartete Tore und die Tipps aller 4 Modelle inklusive eines Konsens-Tipps.

### 3. Einzelne Analysen
Du kannst die Varianten auch separat ausführen:
```bash
python3 prediction_ohne_xG.py
python3 prediction_mit_xG.py
```

---

## 📈 Features
- **Rolling Window:** Statistiken werden immer aus den vorangegangenen 20 Spielen berechnet (kein Data Leakage).
- **Fuzzy Matching:** Automatische Erkennung von Teamnamen bei der Eingabe.
- **Visualisierung:** Aussagekräftige Grafiken für die Bachelorarbeit werden automatisch im PNG-Format erstellt.