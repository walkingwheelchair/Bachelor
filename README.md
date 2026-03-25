# ⚽ Abstract: Vorhersage von Bundesliga-Spielergebnissen – Bachelorarbeit

Dieses Repository enthält den vollständigen Code für eine wissenschaftliche Bachelorarbeit, die untersucht, ob und wie stark die Integration von **Expected Goals (xG)** als Feature die Vorhersagegenauigkeit von Bundesliga-Spielen verbessert. Es vergleicht dabei die Modell-Performance basierend auf reinen Basis-Leistungsdaten mit Modellen, die erweiterte xG-Daten nutzen.

---

## 🔬 Wissenschaftliche Fragestellung & Methodik

Das Ziel des Projekts ist es, den Spielausgang (Heimsieg `H`, Unentschieden `D`, Auswärtssieg `A`) mithilfe von Machine-Learning-Verfahren vorherzusagen.

### Untersuchte Datengrundlage & Feature Engineering
- **Datenbasis:** Historische Bundesliga-Ergebnisse (Ressource: `Bundesliga_Quellen`) und saisonbasierte Expected-Goals-Daten (Ressource: `xG_Quellen`) ab der Saison 2016/17 (bis einschl. 2025/26).
- **Train-Test-Split:** Modelle werden auf Daten der Saisons 2016/17 bis 2022/23 trainiert und auf den folgenden Saisons ab 2023/24 getestet, um reale Voraussagen zu simulieren.
- **Dynamische Formberechnung (EWA):** Statt starren Durchschnitten wird zur Feature-Generierung ein *Exponentially Weighted Average* (EWA)-Verfahren über eine "Rolling Window"-Funktion genutzt. Dabei fließen vergangenheitsbezogene Schüsse, Torschüsse, Punkte sowie erzielte/kassierte Tore gewichtet ein.
- **Verhinderung von Data Leakage bei xG:**
  - Für historische Saisons werden prinzipiell die xG-Werte der beendeten *Vorjahressaison (Lag-1)* verwendet.
  - Für die *aktuell laufende* Saison werden die laufenden Daten genutzt (da die CSVs wöchentlich aktualisiert werden). Dies spiegelt den echten Wissensstand am Spieltag ideal wider.
- **Unentschieden-spezifische Features:** Um das oft schwer vorherzusagende Unentschieden (`D`) besser interpretieren zu können, erzeugt das Modell metrische Gleichgewichts-Indikatoren wie `abs_goal_diff`, `abs_points_diff` und die `combined_draw_tendency`.

### Machine-Learning-Modelle & Techniken
Folgende Algorithmen werden verglichen (jeweils mit und ohne xG-Features):
1. **Logistic Regression** – Referenzmodell mit linearer Entscheidungsfindung.
2. **XGBoost (XGBClassifier)** – Leistungsstarkes Tree-basiertes Ensemble-Verfahren zur Erfassung nicht-linearer Abhängigkeiten.
   - **Hyperparameter-Tuning:** Über eine Option (`DO_TUNING`) in `tuning.py` kann XGBoost systematisch via `RandomizedSearchCV` optimiert werden.
   - **Klassen-Imbalance:** Da Heimsiege deutlich häufiger vorkommen, verwenden beide Logiken ein ausgewogenes Weights-Balancing (`class_weight='balanced'` bzw. `sample_weight`), um Under-Representation abzufangen.

### Evaluierungs-Metriken & Baseline-Abgleich
- **Accuracy & Log-Loss (Cross-Entropy Loss):** Basismetriken zur Bewertung von Genauigkeit und der Kalibrierung der Wahrscheinlichkeiten.
- **Professionelle Baselines:** Die Vorhersagen werden gegen simple "Immer Heimsieg"-Strategien sowie gegen Markt-Wettquoten (`Bet365`) evaluiert.
- **Statistischer Signifikanztest (McNemar-Test):** Zieht die finale wissenschaftliche Konkusion, ob der Austausch zu einer signifikanten Verbesserung der Treffsicherheit führt (p-Wert < 0.05).

---

## 📂 Code-Architektur & Dokumentation

### `utils.py` (Das Herzstück der Datenverarbeitung)
Beinhaltet Core-Funktionen für das Einladen von CSV-Dateien, Matching von abweichenden Teamnamen und führt das entscheidende komplexe Feature-Engineering (EWA, Data-Leakage Behebung, xG-Verarbeitung) durch.

### `prediction_ohne_xG.py` (Baseline-Features)
Baut Vorhersagemodelle *rein auf Basis klassischer Performancedaten*. Trainiert und evaluiert "Logistic Regression" und "XGBoost". Erzeugt Confusion Matrices, Plottet Variable-Importances und speichert alles in `Ergebnisse/`.

### `prediction_mit_xG.py` (Modelle mit Expected Goals)
Führt exakt denselben Ablauf durch, integriert zur Feature-Liste jedoch explizit die Expected Goals Metriken (`xG_per_game`, `xGA_per_game` etc.).

### `tuning.py` (Hyperparameter Optimierung)
Funktionierendes Skript für Cross-Validation Suchräume. Ermöglicht schnelle XGBoost Optimiervorgänge abseits der Hardcodings.

### `vergleich.py` (Wissenschaftliche Auswertung & Synthese)
Das zentrale Evaluations-Skript der Arbeit. Ruft die Modelle ab, führt den Baseline/Signifikanz-Vergleich durch, verarbeitet die Confusion-Matrices aller vier Modelle zu einem 2x2 Grid, erstellt Accuracy/Log-Loss Balkendiagramme sowie Radar-Charts (Precision/Recall) zur Identifizierung von Bias. 

### `predict_spiel.py` (Interaktive Live-Anwendung)
Nutzt die aus der Analyse gewonnenen Algorithmen für Vorhersagen kommender Partien als Kommandozeilen-Tool.
- **Modell-Persistenz:** Dank `joblib`-Caching werden Trainingsläufe im Ordner `model_cache/` intelligent gesichert. Sind die Trainings-Saisons unverändert, überspringt das Skript zeitintensive Retrainings und ist sofort nutzbar.
- Berechnet dynamisch die aktuelle EWA-Formmerkmale, liefert Grobschätzungen und gibt am Ende einen demokratischen Konsens aller 4 Vorhersagen für Tippzwecke zurück.

---

## 🚀 Installation & Ausführung

Stelle zunächst sicher, dass Python (>=3.9) installiert ist. Danach benötigte Bibliotheken installieren:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost statsmodels joblib
```

**Zwei zentrale Anwendungsbeispiele:**

1. **Thesis-Analyse berechnen & sämtliche Diagramme generieren:**
   ```bash
   python3 vergleich.py
   ```
   *(Erstellt alle Vergleiche, Hypothesentests und Grafiken im Ordner `Ergebnisse/`)*

2. **Interaktives Vorhersage-Terminal für anstehende Spieltage starten:**
   ```bash
   python3 predict_spiel.py
   ```
   *(Erlaubt die schnelle Eingabe wie "Bayern" vs. "Bremen" dank Modell-Caching in Millisekunden).*