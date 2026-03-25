# ⚽ Abstract: Vorhersage von Bundesliga-Spielergebnissen – Bachelorarbeit

Dieses Repository enthält den vollständigen Code für eine wissenschaftliche Bachelorarbeit, die untersucht, ob und wie stark die Integration von **Expected Goals (xG)** als Feature die Vorhersagegenauigkeit von Bundesliga-Spielen verbessert. Es vergleicht dabei die Modell-Performance basierend auf reinen Basis-Leistungsdaten mit Modellen, die erweiterte xG-Daten nutzen.

---

## 🔬 Wissenschaftliche Fragestellung & Methodik

Das Ziel des Projekts ist es, den Spielausgang (Heimsieg `H`, Unentschieden `D`, Auswärtssieg `A`) mithilfe von Machine-Learning-Verfahren vorherzusagen.

### Untersuchte Datengrundlage & Feature Engineering
- **Datenbasis:** Historische Bundesliga-Ergebnisse (Ressource: `Bundesliga_Quellen`) und saisonbasierte Expected-Goals-Daten (Ressource: `xG_Quellen`) ab der Saison 2016/17 (bis einschl. 2025/26).
- **Train-Test-Split:** Modelle werden auf Daten der Saisons 2016/17 bis 2022/23 trainiert und auf den folgenden Saisons 2023/24 bis 2025/26 getestet, um reale Voraussagen zu simulieren.
- **Dynamische Formberechnung (EWA):** Statt starren Durchschnitten wird zur Feature-Generierung ein  *Exponentially Weighted Average* (EWA)-Verfahren über eine "Rolling Window"-Funktion genutzt. Dabei fließen vergangenheitsbezogene Schüsse, Torschüsse, Punkte sowie erzielte/kassierte Tore in der Heimbilanz und Auswärtsbilanz gewichtet (mit abnehmender Relevanz älterer Spiele) ein. Data Leakage wird hierdurch strikt vermieden.

### Machine-Learning-Modelle
Folgende Algorithmen werden miteinander verglichen (jeweils mit und ohne xG-Features):
1. **Logistic Regression** – Referenzmodell mit linearer Entscheidungsfindung.
2. **XGBoost (XGBClassifier)** – Leistungsstarkes Tree-basiertes Ensemble-Verfahren zur Erfassung nicht-linearer Abhängigkeiten.

### Evaluierungs-Metriken
Um den Nutzen der xG-Daten wissenschaftlich zu messen, generieren die Skripte tiefgehende Metriken:
- **Accuracy:** Anteil der korrekt vorhergesagten Spielausgänge an allen Spielen.
- **Log-Loss (Cross-Entropy Loss):** Maß für die Verlässlichkeit und Kalibrierung der Modell-Wahrscheinlichkeiten.
- **Precision & Recall:** Identifikation der klassenspezifischen Vorhersagequalität (via Radar-Charts).

---

## 📂 Code-Architektur & Dokumentation

Das Repository stellt verschiedene ausführbare Skripte zur Verfügung, die exakt vordefinierte Schritte der Machine-Learning-Pipeline abbilden:

### `utils.py` (Das Herzstück der Datenverarbeitung)
- **Was es macht:** Beinhaltet Core-Funktionen für das Einladen von CSV-Dateien (`load_bundesliga_data`, `load_xg_data`) und Zusammenführen der Datensätze.
- **Vorgehen:** Nimmt ein "Fuzzy-Matching" bei Team-Namen vor (da Schreibweisen zwischen Datensätzen differieren können) und führt das entscheidende Feature-Engineering durch: Die Generierung der dynamischen EWA-Formdaten, Hinzufügen der xG-Referenzen pro Match, sowie die Berechnung von Differenz-Features (z.B. Tordifferenz).

### `prediction_ohne_xG.py` (Baseline-Modelle)
- **Was es macht:** Baut Vorhersagemodelle *rein auf Basis klassischer Performancedaten* (vergangene Tore, Punkte, Schüsse).
- **Vorgehen:** Trainiert und evaluiert "Logistic Regression" und "XGBoost". Erzeugt Confusion Matrices (`confusion_ohne_xG.png`), plottet die "Top-20 Feature Importance" des XGBoost Modells und speichert aggregierte Kennzahlen in einer CSV innerhalb des automatisch erstellten Ordners `Ergebnisse/`.

### `prediction_mit_xG.py` (Modelle mit Expected Goals)
- **Was es macht:** Führt exakt denselben Ablauf durch, integriert zur Feature-Liste jedoch zusätzlich die Expected Goals Metriken (`xG_per_game`, `xGA_per_game`, `xPTS_per_game`) beider Teams.
- **Vorgehen:** Trainiert erweiterte Varianten der Modelle auf den neuen Features, sodass direkte Vergleiche gezogen werden können. Output wieder in `Ergebnisse/`.

### `vergleich.py` (Wissenschaftliche Auswertung & Synthese)
- **Was es macht:** Das zentrale Evaluations-Skript der Arbeit. Es ruft intern die Funktionen aus der *Ohne xG*- und *Mit xG*-Kalkulation ab, vereint sie und misst den finalen "Expected Goals"-Effekt.
- **Vorgehen:** Stellt alle 4 Modell-Resultate gegenüber und generiert aussagekräftige Graphiken für die Thesis: Balkendiagramme (Accuracy & Log-Loss Nebeneinanderstellung), ein Radar-Chart für Precision/Recall und ein 2x2 Plot aller Confusion Matrices. Im Terminal gibt es zudem eine Summary-Tabelle aus, analysiert signifikante Verbesserung/Verschlechterung und kürt objektiv den Sieger.

### `predict_spiel.py` (Interaktive Live-Anwendung)
- **Was es macht:** Nutzt die aus der Analyse gewonnenen Algorithmen für Vorhersagen kommender Partien. Konzeptioniert als ein "Live-Wett-Tool".
- **Vorgehen:** Nach Programmstart trainiert das Skript dynamisch alle 4 Algorithmen parallel auf *sämtlichen* vorhandenen historischen Spieldaten im Hintergrund. Lässt den User anschließend völlig interaktiv (in Endlosschleife) Paarungen (Heim & Auswärts) in die Kommandozeile eingeben.
- **Output:** Es berechnet On-The-Fly die aktuellen EWA-Formmerkmale für das bevorstehende Spiel, liefert dem User Form-Statistiken, eine "grobe Torschätzung" für das Spiel und wirft anschließend für jedes Modell die exakten Match-Wahrscheinlichkeiten für Heimsieg, Unentschieden und Auswärtssieg aus. Am Ende bildet es einen demokratischen "Konsens-Tipp" der Modelle.

---

## 🚀 Installation & Ausführung

Stelle zunächst sicher, dass Python (>=3.9) installiert ist. Danach benötigte Bibliotheken referenzieren:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
```

**Zwei zentrale Anwendungsbeispiele:**

1. **Thesis-Analyse berechnen & sämtliche Diagramme generieren:**
   ```bash
   python3 vergleich.py
   ```
   *(Erstellt alle vergleichenden Visualisierungen automatisiert im Ordner `Ergebnisse/`)*

2. **Interaktives Vorhersage-Terminal für anstehende Spieltage starten:**
   ```bash
   python3 predict_spiel.py
   ```
   *(Erlaubt die Konsoleneingabe wie beispielsweise "Bayern" vs. "Dortmund")*