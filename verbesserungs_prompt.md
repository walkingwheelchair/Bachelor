# Prompt: Verbesserung des Bundesliga-Vorhersage-Projekts (Bachelorarbeit)

## Kontext & Projektübersicht

Du arbeitest an einer bestehenden Python-Codebasis für eine Bachelorarbeit zum Thema **"Vorhersage von Bundesliga-Spielergebnissen mit Machine Learning"**. Das Projekt vergleicht vier Modelle (Logistic Regression & XGBoost, jeweils mit und ohne Expected-Goals-Features) und sagt Spielausgänge als Drei-Klassen-Problem vorher (H = Heimsieg, D = Unentschieden, A = Auswärtssieg).

Die Projektstruktur besteht aus folgenden Dateien:
- `utils.py` – Datenladen, Team-Mapping, EWA-Feature-Engineering
- `prediction_ohne_xG.py` – Baseline-Modelle ohne xG
- `prediction_mit_xG.py` – Modelle mit xG-Features
- `vergleich.py` – Vergleichs-Plots und Zusammenfassung
- `predict_spiel.py` – Interaktives Live-Vorhersage-Tool
- `Bundesliga_Quellen/` – CSV-Spieldaten (Saison 2016/17 bis 2025/26)
- `xG_Quellen/` – CSV-Saisondaten mit xG, xGA, xPTS pro Team

---

## Aufgabe: Implementiere die folgenden 7 Verbesserungen

Bitte implementiere **alle 7 Verbesserungen vollständig** in den jeweiligen Dateien. Erkläre nach jeder Änderung kurz, was du geändert hast und warum. Verändere dabei die bestehende Projektstruktur und Logik nicht unnötig – ergänze und verbessere nur gezielt.

---

### Verbesserung 1: xG-Data-Leakage beheben (höchste Priorität)

**Problem:**
Die xG-Daten (`xG_per_game`, `xGA_per_game`, `xPTS_per_game`) werden aktuell als **Saisondurchschnitt** aus den `xG_Quellen/`-CSVs gemergt. Das bedeutet, dass am 1. Spieltag einer Saison das Modell bereits den xG-Wert kennt, der erst am Ende der Saison feststeht. Das ist konzeptuell unehrlich – besonders beim interaktiven `predict_spiel.py` für laufende Saisons.

**Gewünschte Lösung:**
- Berechne xG ebenfalls als **Rolling-EWA über vergangene Spiele**, analog zur bestehenden EWA-Berechnung für Tore, Punkte und Schüsse in `compute_rolling_features()`.
- Da die `xG_Quellen/`-CSVs nur **Saisonsummen** enthalten (keine Match-Level-xG), ist der pragmatische Ansatz folgender:
  - Nutze die Saisondaten aus dem **Vorjahr** als xG-Prior für das aktuelle Spieljahr. Merge also für jede Saison die xG-Werte der *vorherigen* Saison (Season lag-1) statt der aktuellen.
  - Das entspricht der realistischen Situation: Vor Saisonbeginn kennt man die Stärke eines Teams aus der letzten Saison.
- Passe `merge_xg_features()` in `utils.py` entsprechend an.
- Dokumentiere diesen Ansatz mit einem Kommentar im Code und weise in den Print-Ausgaben darauf hin (z.B. `"📊 xG-Daten: Vorjahreswerte als Prior genutzt (kein Leakage)"`).

---

### Verbesserung 2: Hyperparameter-Tuning für XGBoost

**Problem:**
XGBoost läuft mit fest hardgecodierten Parametern (`n_estimators=300, max_depth=6, learning_rate=0.08`), die nie systematisch optimiert wurden.

**Gewünschte Lösung:**
- Füge in `utils.py` oder einer neuen Datei `tuning.py` eine Funktion `tune_xgboost(X_train, y_train)` hinzu, die mit **`RandomizedSearchCV`** aus scikit-learn einen Hyperparameter-Search durchführt.
- Suchraum:
  ```python
  param_dist = {
      "clf__n_estimators":  [100, 200, 300, 400],
      "clf__max_depth":     [3, 4, 5, 6, 7],
      "clf__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
      "clf__subsample":     [0.7, 0.8, 0.9, 1.0],
      "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
      "clf__min_child_weight":  [1, 3, 5],
  }
  ```
- Nutze `cv=3`, `n_iter=30`, `scoring="accuracy"`, `random_state=42`.
- Die Funktion soll die **besten Parameter ausgeben** und das optimierte Modell zurückgeben.
- Integriere das Tuning in `prediction_mit_xG.py` und `prediction_ohne_xG.py` – aber nur wenn eine Konstante `DO_TUNING = False` am Dateianfang auf `True` gesetzt ist, damit der normale Durchlauf schnell bleibt.
- Gib nach dem Tuning die besten Parameter und die Cross-Validation-Accuracy aus.

---

### Verbesserung 3: Statistischer Signifikanztest (McNemar-Test)

**Problem:**
Die Verbesserung durch xG beträgt bei Logistic Regression +2,94 Prozentpunkte. Es wird jedoch nie geprüft, ob dieser Unterschied **statistisch signifikant** ist oder ob er zufällig entstanden sein könnte.

**Gewünschte Lösung:**
- Füge in `vergleich.py` eine Funktion `mcnemar_test(result_ohne, result_mit)` hinzu.
- Implementiere den **McNemar-Test** mit `statsmodels.stats.contingency_tables.mcnemar` (oder manuell über scipy falls statsmodels nicht verfügbar).
- Die Kontingenztabelle vergleicht jeweils die Vorhersagen von Modell-OHNE-xG vs. Modell-MIT-xG auf denselben Testdaten: Wie viele Spiele sagt nur das eine richtig, wie viele nur das andere?
- Berechne den Test für beide Modellpaare (Logistic Regression ohne vs. mit xG und XGBoost ohne vs. mit xG).
- Gib aus:
  ```
  McNemar-Test: Logistic Regression (OHNE vs. MIT xG)
    Korrekt nur OHNE xG: X Spiele
    Korrekt nur MIT xG:  Y Spiele
    Chi²: Z.ZZ  |  p-Wert: 0.XXX
    → Unterschied ist [statistisch signifikant (p < 0.05) / NICHT signifikant]
  ```
- Rufe diese Funktion am Ende von `print_summary_table()` auf.

---

### Verbesserung 4: Baseline-Vergleiche ergänzen

**Problem:**
Als einzige Baseline existiert die 1/3-Zufallslinie im Accuracy-Plot. Ein sinnvoller Vergleich mit realistischeren Baselines fehlt.

**Gewünschte Lösung:**
Füge in `vergleich.py` eine Funktion `compute_baselines(y_test, df_test)` hinzu, die zwei Baselines berechnet:

**Baseline 1 – "Immer Heimsieg":**
- Predict immer `"H"` für alle Spiele.
- Berechne Accuracy und Log-Loss (verwende für Log-Loss Wahrscheinlichkeiten `[1.0, 0.0, 0.0]` für H, D, A).

**Baseline 2 – Wettquoten-Baseline:**
- Die Bundesliga-CSVs enthalten Wettquoten (z.B. `B365H`, `B365D`, `B365A` von Bet365).
- Konvertiere Quoten in Wahrscheinlichkeiten: `p_H = 1/B365H`, dann normalisieren auf Summe 1 (Vig entfernen).
- Wähle als Vorhersage die Klasse mit der höchsten Wahrscheinlichkeit.
- Berechne Accuracy und Log-Loss.
- Falls keine Wettquoten-Spalten in den Testdaten vorhanden sind, überspringe diese Baseline mit einem Hinweis.

Ergänze beide Baselines in `plot_accuracy_logloss()` als **gestrichelte horizontale Linien** im Accuracy-Subplot (ähnlich der bestehenden 1/3-Linie), beschriftet mit `"Baseline: Immer Heimsieg"` und `"Baseline: Wettquoten"`.

Gib die Baseline-Werte außerdem in der Terminal-Zusammenfassung aus.

---

### Verbesserung 5: Klassen-Imbalance adressieren

**Problem:**
Heimsiege (~45%), Auswärtssiege (~30%) und Unentschieden (~25%) sind ungleich verteilt. Die Modelle verwenden kein Class-Weighting, was die schlechte Performance bei Unentschieden erklärt.

**Gewünschte Lösung:**
- Füge in `prediction_ohne_xG.py` und `prediction_mit_xG.py` bei der Logistic Regression den Parameter `class_weight='balanced'` hinzu.
- Füge bei XGBoost eine manuelle Gewichtung über den `sample_weight`-Parameter in `.fit()` hinzu:
  ```python
  from sklearn.utils.class_weight import compute_sample_weight
  sample_weights = compute_sample_weight("balanced", y_train)
  model.fit(X_train, y_train_enc, clf__sample_weight=sample_weights)
  ```
- Drucke nach dem Evaluation-Report einen Hinweis aus: `"ℹ️  Class-Weighting aktiv: balanced"`.
- Prüfe und kommentiere im Code, ob sich die Recall-Werte für Unentschieden verbessert haben.

---

### Verbesserung 6: Modellpersistenz in `predict_spiel.py`

**Problem:**
`predict_spiel.py` trainiert bei jedem Programmstart alle 4 Modelle neu – das dauert ca. 30–60 Sekunden und ist unnötig, wenn sich die Datenbasis nicht geändert hat.

**Gewünschte Lösung:**
- Nutze `joblib` zum Speichern und Laden der trainierten Modelle.
- Definiere einen Cache-Ordner `model_cache/` im Projektverzeichnis.
- Speichere nach dem Training folgende Dateien:
  - `model_cache/models_ohne_xG.joblib`
  - `model_cache/models_mit_xG.joblib`
  - `model_cache/label_encoder.joblib`
  - `model_cache/meta.json` – enthält den Timestamp der letzten Trainingsdatei (prüfe via `os.path.getmtime` auf alle CSVs in `Bundesliga_Quellen/` und `xG_Quellen/`)
- Beim Programmstart: Prüfe ob der Cache existiert **und** ob keine CSV neuer ist als der Cache-Timestamp.
  - Wenn Cache gültig: Lade Modelle aus Cache (`"⚡ Modelle aus Cache geladen (kein Re-Training nötig)"`)
  - Wenn Cache veraltet oder nicht vorhanden: Trainiere neu und speichere Cache
- Füge `model_cache/` zur `.gitignore` hinzu (oder erstelle sie, falls nicht vorhanden).

---

### Verbesserung 7: Unentschieden-spezifische Features

**Problem:**
Das Modell hat keine Features, die gezielt das **Gleichgewicht** zwischen zwei Teams messen – was aber entscheidend für die Vorhersage von Unentschieden ist.

**Gewünschte Lösung:**
Füge in `utils.py` in der Funktion `add_derived_features()` folgende neue Features hinzu:

```python
# Absolutwerte der Differenzen (je kleiner, desto ausgeglichener das Spiel)
df["abs_goal_diff"]   = df["goal_diff_avg"].abs()
df["abs_points_diff"] = df["points_diff_avg"].abs()

# Formstabilität: Standardabweichung der letzten Ergebnisse (Proxy über EWA-Punkte)
# Je niedriger die Punkte beider Teams (nahe 1.0/Spiel), desto wahrscheinlicher Unentschieden
df["combined_draw_tendency"] = (
    1.0 / (1.0 + df["abs_goal_diff"]) *
    1.0 / (1.0 + df["abs_points_diff"])
)
```

- Füge diese 3 neuen Features (`abs_goal_diff`, `abs_points_diff`, `combined_draw_tendency`) zu `BASE_FEATURES` und `XG_FEATURES` in `utils.py` hinzu.
- Ergänze bei den xG-Features außerdem:
  ```python
  df["abs_xG_diff"] = df["xG_diff"].abs()
  ```
  und füge `"abs_xG_diff"` zu `XG_FEATURES` hinzu.
- Kommentiere die neuen Features im Code mit `# [NEU] Unentschieden-Features`.

---

## Abschließende Anforderungen

- **Kompatibilität:** Alle Änderungen müssen rückwärtskompatibel sein. `vergleich.py` und `predict_spiel.py` sollen weiterhin ohne Anpassungen funktionieren.
- **Ausgabe:** Jede Verbesserung soll beim Ausführen durch eine kurze Terminal-Meldung sichtbar sein (z.B. `"✅ [Verbesserung 3] McNemar-Test aktiv"`).
- **Keine neuen Abhängigkeiten außer:** `statsmodels`, `joblib` (beide typischerweise bereits installiert mit scikit-learn/scipy).
- **Code-Stil:** Halte den bestehenden Stil bei (deutsche Kommentare, Emoji-Prints, klare Abschnittstrennungen mit `# ─────`).
- **Reihenfolge:** Implementiere die Verbesserungen in der Reihenfolge 7 → 5 → 4 → 3 → 6 → 2 → 1, da spätere Verbesserungen auf früheren aufbauen können.

Beginne mit einer kurzen Zusammenfassung, welche Dateien du anfassen wirst, bevor du mit der Implementierung startest.




In utils.py in der Funktion merge_xg_features() soll die Logik für den Saison-Lag angepasst werden. Aktuell wird für alle Saisons der Vorjahres-xG-Wert genutzt. Das soll so geändert werden:

Für abgeschlossene Saisons (alles außer der aktuellen) bleibt der Vorjahres-Lag wie er ist.
Für die aktuelle Saison (also die, die in den xG-Daten als maximale Saison vorkommt) werden die xG-Werte direkt ohne Lag genutzt – weil diese CSV wöchentlich aktuell gehalten wird und damit kein Leakage entsteht.

Die aktuelle Saison soll dynamisch ermittelt werden via xg_df["Season"].max(), nicht hardgecoded.
