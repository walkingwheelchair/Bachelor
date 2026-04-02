# Erklärung aller Kennzahlen im Notebook

## Überblick

Das Notebook baut eine Machine-Learning-Pipeline zur Vorhersage von Bundesliga-Spielergebnissen. Es gibt zwei Featuremengen: **BASE_FEATURES** (ohne xG-Daten, 25 Features) und **XG_FEATURES** (mit xG-Daten als Erweiterung, 34 Features). Alle Features werden für je zwei Teams pro Spiel berechnet: dem Heimteam (`home_team_*`) und dem Auswärtsteam (`away_team_*`).

---

## Zielvariable: FTR (Full Time Result)

Dies ist das, was das Modell vorhersagen soll – das Spielergebnis:

- **H** = Heimsieg
- **D** = Unentschieden (Draw)
- **A** = Auswärtssieg (Away Win)

Für die Korrelationsanalyse wird FTR numerisch kodiert:
- H → **+1** (Erfolg Heimteam)
- D → **0** (Unentschieden)
- A → **−1** (Niederlage Heimteam)

---

## Was ist der EWA (Exponentially Weighted Average)?

Das Herzstück aller Kennzahlen ist der **EWA – exponentiell gewichteter Durchschnitt**. Er berechnet einen gleitenden Durchschnitt über alle bisherigen Spiele einer Mannschaft in einer Saison, wobei **neuere Spiele stärker gewichtet** werden als ältere.

Der Unterschied zu einem normalen Durchschnitt (z.B. "letzten 5 Spiele"):
- **EWA:** Alle Spiele fließen ein, aber ältere zählen weniger
- **Normaler Durchschnitt:** Nur die letzten N Spiele zählen, alle gleich

> **Formel:** `EWA_neu = α × aktueller_Wert + (1 − α) × EWA_alt`
> mit `α = 2 / (span + 1)`. Je größer der Span, desto träger/glatter der Durchschnitt.

**Beispiel:** Wenn ein Team plötzlich viele Tore schießt, steigt der EWA sofort an – aber alle früheren Leistungen bleiben im Hintergrund spürbar.

---

## BASE_FEATURES – Die 25 Basis-Kennzahlen

### Gruppe 1: Gesamtform (overall) – gilt für alle Spiele (Heim + Auswärts)

Diese 10 Features beschreiben, wie gut eine Mannschaft **insgesamt** (egal ob Heim oder Auswärts) in der laufenden Saison gespielt hat:

| Feature | Beschreibung |
|---------|-------------|
| `home_team_overall_ewa_GoalsScored` | EWA der **erzielten Tore** des Heimteams (alle Spiele) |
| `home_team_overall_ewa_GoalsConceded` | EWA der **kassierten Tore** des Heimteams (alle Spiele) |
| `home_team_overall_ewa_Points` | EWA der **Punkte** des Heimteams (3=Sieg, 1=Unentschieden, 0=Niederlage) |
| `home_team_overall_ewa_Shots` | EWA der **abgegebenen Schüsse** des Heimteams |
| `home_team_overall_ewa_ShotsOnTarget` | EWA der **Schüsse aufs Tor** des Heimteams |
| `away_team_overall_ewa_GoalsScored` | EWA der **erzielten Tore** des Auswärtsteams (alle Spiele) |
| `away_team_overall_ewa_GoalsConceded` | EWA der **kassierten Tore** des Auswärtsteams (alle Spiele) |
| `away_team_overall_ewa_Points` | EWA der **Punkte** des Auswärtsteams (alle Spiele) |
| `away_team_overall_ewa_Shots` | EWA der **abgegebenen Schüsse** des Auswärtsteams |
| `away_team_overall_ewa_ShotsOnTarget` | EWA der **Schüsse aufs Tor** des Auswärtsteams |

### Gruppe 2: Heimform (home_form) – nur Heimspiele des Heimteams

Diese 5 Features beschreiben, wie gut das Heimteam **speziell in Heimspielen** performt:

| Feature | Beschreibung |
|---------|-------------|
| `home_team_home_form_ewa_GoalsScored` | EWA der erzielten Tore des Heimteams **nur in Heimspielen** |
| `home_team_home_form_ewa_GoalsConceded` | EWA der kassierten Tore des Heimteams **nur in Heimspielen** |
| `home_team_home_form_ewa_Points` | EWA der Punkte des Heimteams **nur in Heimspielen** |
| `home_team_home_form_ewa_Shots` | EWA der Schüsse des Heimteams **nur in Heimspielen** |
| `home_team_home_form_ewa_ShotsOnTarget` | EWA der Schüsse aufs Tor des Heimteams **nur in Heimspielen** |

### Gruppe 3: Auswärtsform (away_form) – nur Auswärtsspiele des Auswärtsteams

Diese 5 Features beschreiben, wie gut das Auswärtsteam **speziell in Auswärtsspielen** performt:

| Feature | Beschreibung |
|---------|-------------|
| `away_team_away_form_ewa_GoalsScored` | EWA der erzielten Tore des Auswärtsteams **nur in Auswärtsspielen** |
| `away_team_away_form_ewa_GoalsConceded` | EWA der kassierten Tore des Auswärtsteams **nur in Auswärtsspielen** |
| `away_team_away_form_ewa_Points` | EWA der Punkte des Auswärtsteams **nur in Auswärtsspielen** |
| `away_team_away_form_ewa_Shots` | EWA der Schüsse des Auswärtsteams **nur in Auswärtsspielen** |
| `away_team_away_form_ewa_ShotsOnTarget` | EWA der Schüsse aufs Tor des Auswärtsteams **nur in Auswärtsspielen** |

> **Warum Heim- und Auswärtsform getrennt?** Manche Teams sind zuhause ein Schwergewicht, auswärts aber viel schwächer. Diese getrennten Features erlauben dem Modell, den **Heimvorteil** zu berücksichtigen.

### Gruppe 4: Differenz-Features (Stärkenvergleich)

Diese 2 Features vergleichen direkt die Stärke beider Teams:

| Feature | Beschreibung |
|---------|-------------|
| `goal_diff_avg` | `home_overall_GoalsScored − away_overall_GoalsScored` – positiv = Heimteam schießt mehr Tore |
| `points_diff_avg` | `home_overall_Points − away_overall_Points` – positiv = Heimteam hat mehr Punkte gesammelt |

### Gruppe 5: Unentschieden-Features (neu hinzugefügt)

Diese 3 Features helfen dem Modell, **Unentschieden** besser vorherzusagen – eine der schwierigsten Klassen:

| Feature | Beschreibung |
|---------|-------------|
| `abs_goal_diff` | `|goal_diff_avg|` – Absolutwert der Tordifferenz. **Je kleiner, desto ausgeglichener** das Spiel |
| `abs_points_diff` | `|points_diff_avg|` – Absolutwert der Punktedifferenz. **Je kleiner, desto ausgeglichener** |
| `combined_draw_tendency` | `1/(1 + abs_goal_diff) × 1/(1 + abs_points_diff)` – **Kombinierter Unentschieden-Indikator**: nahe 1.0 = beide Teams völlig gleichwertig = Unentschieden wahrscheinlicher |

---

## XG_FEATURES – Zusätzliche 9 xG-Kennzahlen

**xG (Expected Goals = erwartete Tore)** ist eine moderne Fußballstatistik, die misst, wie viele Tore eine Mannschaft basierend auf der **Qualität ihrer Chancen** theoretisch hätte erzielen sollen – unabhängig von Glück, einem guten/schlechten Torwart oder dem berühmten Aluminium.

Die xG-Daten kommen saisonweise als Durchschnittswerte pro Spiel. Um **Data-Leakage** zu vermeiden (d.h. keine Informationen aus der Zukunft zu benutzen), gilt:
- Für **abgeschlossene Saisons:** Wird der xG-Wert der **Vorsaison** als Schätzwert genutzt
- Für die **aktuelle Saison:** Werden die aktuellen Daten direkt genutzt (da sie wöchentlich aktualisiert werden)

| Feature | Beschreibung |
|---------|-------------|
| `home_xG_per_game` | Durchschnittlich **erwartete eigene Tore** (Angriffsstärke) des Heimteams pro Spiel |
| `home_xGA_per_game` | Durchschnittlich **erwartete kassierte Tore** (Defensivschwäche) des Heimteams pro Spiel |
| `home_xPTS_per_game` | Durchschnittlich **erwartete Punkte** des Heimteams pro Spiel (berechnet aus xG-Werten) |
| `away_xG_per_game` | Durchschnittlich **erwartete eigene Tore** des Auswärtsteams pro Spiel |
| `away_xGA_per_game` | Durchschnittlich **erwartete kassierte Tore** des Auswärtsteams pro Spiel |
| `away_xPTS_per_game` | Durchschnittlich **erwartete Punkte** des Auswärtsteams pro Spiel |
| `xG_diff` | `home_xG_per_game − away_xG_per_game` – positiv = Heimteam ist offensiv stärker |
| `xGA_diff` | `home_xGA_per_game − away_xGA_per_game` – positiv = Heimteam ist defensiv schwächer |
| `abs_xG_diff` | `|xG_diff|` – Absolutwert der xG-Differenz; je kleiner, desto ausgeglichener (Unentschieden-Feature) |

---

## Die 8 Datensatz-Varianten

Das Notebook vergleicht 8 Kombination aus:
1. **Ausreißer-Bereinigung:** Mit oder ohne Entfernung von Ausreißern (z.B. extreme Ergebnisse wie 8:0)
2. **xG-Daten:** Mit oder ohne Expected Goals Statistiken
3. **Feature-Selektion:** Alle Features vs. nur die stärksten (hoch korrelierenden) Features

| Variante | Ausreißer entfernt | xG-Daten | Feature-Set |
|----------|--------------------|----------|-------------|
| df1 | Nein | Nein | BASE_FEATURES (25) |
| df2 | Nein | Ja | XG_FEATURES (34) |
| df3 | Nein | Nein | Selektierte Features |
| df4 | Nein | Ja | Selektierte Features + xG |
| df5 | **Ja** | Nein | BASE_FEATURES (25) |
| df6 | **Ja** | Ja | XG_FEATURES (34) |
| df7 | **Ja** | Nein | Selektierte Features |
| df8 | **Ja** | Ja | Selektierte Features + xG |

---

## Modell & Evaluationsmetriken

- **Modell:** XGBoost Klassifikator (Gradient Boosting)
- **Hyperparameter-Tuning:** RandomizedSearchCV mit StratifiedKFold (5 Folds)
- **Trainingsset:** Saison 2023–2024
- **Testset:** Saison 2024–2025 / 2025–2026
- **Klassen:** H (Heimsieg), D (Unentschieden), A (Auswärtssieg)

---

## Die Evaluationsmetriken – einfach und mathematisch erklärt

Da wir ein **3-Klassen-Klassifikationsproblem** haben (H / D / A), wird jede Metrik
entweder global über alle Klassen oder **pro Klasse** berechnet.

---

### 1. Konfusionsmatrix (Confusion Matrix)

Die Konfusionsmatrix ist die Grundlage für alle anderen Metriken. Sie zeigt,
**wie oft das Modell was vorhergesagt hat – und was tatsächlich richtig war**.

```
               Vorhergesagt
               H      D      A
           ┌──────┬──────┬──────┐
           │      │      │      │
Tatsächl. H│  TP  │  FN  │  FN  │
           │      │      │      │
          D│  FP  │  TP  │  FN  │
           │      │      │      │
          A│  FP  │  FP  │  TP  │
           └──────┴──────┴──────┘
```

Die Diagonale (TP) = alles richtig vorhergesagt.
Alles außerhalb = Fehler des Modells.

Für jede Klasse wird „One vs. Rest" gewertet:

- **TP (True Positive)**  = Modell sagt H → tatsächlich H  ✅
- **FP (False Positive)** = Modell sagt H → aber war D oder A  ❌
- **FN (False Negative)** = Modell sagt nicht H → aber war wirklich H  ❌
- **TN (True Negative)**  = Modell sagt nicht H → und war auch kein H  ✅

---

### 2. Accuracy (Trefferquote)

**Was sie misst:** Wie viel Prozent aller Spiele wurden korrekt vorhergesagt?

**Mathematisch:**

```
             Anzahl korrekt vorhergesagter Spiele
Accuracy =  ─────────────────────────────────────
                 Gesamtanzahl aller Spiele


       TP_H + TP_D + TP_A
Acc = ──────────────────────
            N_gesamt
```

**Beispiel:** 100 Spiele, 52 korrekt → Accuracy = 52%

> **Schwäche:** Bei ungleichen Klassengrößen kann ein Modell hohe Accuracy
> erreichen, indem es einfach immer „H" (die häufigste Klasse) vorhersagt.
> Deshalb brauchen wir Precision, Recall und F1.

---

### 3. Precision (Genauigkeit)

**Was sie misst:** Wenn das Modell „H" vorhersagt – wie oft stimmt das wirklich?
→ *Wie zuverlässig sind die positiven Vorhersagen?*

**Mathematisch (pro Klasse):**

```
               TP
Precision = ────────
             TP + FP
```

**Beispiel für Klasse H:**
- Modell sagt 40× „H"
- Davon wirklich Heimsiege: 28 → **Precision(H) = 28/40 = 0.70**
- 12× falsch getippt (FP)

> **Niedrige Precision** = Modell ist zu optimistisch, tippt zu oft H

---

### 4. Recall (Trefferrate / Sensitivität)

**Was sie misst:** Von allen tatsächlichen Heimsiegen – wie viele hat das Modell gefunden?
→ *Wie vollständig ist das Modell?*

**Mathematisch (pro Klasse):**

```
            TP
Recall = ────────
          TP + FN
```

**Beispiel für Klasse H:**
- 50 echte Heimsiege im Testset
- Modell hat 28 davon erkannt → **Recall(H) = 28/50 = 0.56**
- 22 Heimsiege wurden als D oder A vorhergesagt (FN)

> **Niedriger Recall** = Modell übersieht viele echte Heimsiege

**Precision vs. Recall – der Zielkonflikt:**

| Modellverhalten | Precision | Recall |
|----------------|-----------|--------|
| Tippt selten „H" (vorsichtig) | Hoch (wenige Fehler) | Niedrig (viele übersehen) |
| Tippt oft „H" (optimistisch) | Niedrig (viele Fehler) | Hoch (wenige übersehen) |

---

### 5. F1-Score

**Was er misst:** Der F1-Score balanciert Precision und Recall in **einer einzigen Zahl**.
→ *Nützlich, wenn man keinen der beiden Aspekte einseitig bevorzugen will.*

**Mathematisch (pro Klasse):**

```
             2 × Precision × Recall
F1 = ─────────────────────────────────
           Precision + Recall


        2 × TP
F1 = ───────────────────
      2×TP + FP + FN
```

Der F1 ist das **harmonische Mittel** – es bestraft sehr stark, wenn eine
der beiden Größen nahe Null ist:

| Precision | Recall | F1    |
|-----------|--------|-------|
| 1.00      | 0.00   | 0.00  |
| 0.70      | 0.70   | 0.70  |
| 0.90      | 0.50   | 0.643 |
| 0.70      | 0.56   | 0.622 |

**Beispiel:** Precision(H)=0.70, Recall(H)=0.56
→ F1(H) = 2 × 0.70 × 0.56 / (0.70 + 0.56) = 0.784 / 1.26 ≈ **0.62**

---

### 6. Macro-Average vs. Weighted-Average

Der Classification Report aggregiert diese Metriken über alle 3 Klassen:

**Macro-Average:** Einfacher Durchschnitt – alle Klassen gleich gewichtet

```
Macro-F1 = (F1_H + F1_D + F1_A) / 3
```

→ Behandelt Unentschieden (seltener) genauso wichtig wie Heimsiege.

**Weighted-Average:** Gewichteter Durchschnitt – häufigere Klassen zählen mehr

```
                F1_H × n_H + F1_D × n_D + F1_A × n_A
Weighted-F1 = ──────────────────────────────────────────
                            N_gesamt
```

→ Spiegelt realistischer wider, wie das Modell in der Praxis abschneidet.

---

### 7. Überblick – Welche Metrik bedeutet was?

| Metrik | Frage | Gut wenn... |
|--------|-------|-------------|
| **Accuracy** | Wie oft richtig insgesamt? | > 50% (Zufallsbaseline liegt bei ~40–45%) |
| **Precision** | Wenn ich H tippe, stimmt das? | Hoch → wenige Fehlalarme |
| **Recall** | Finde ich alle H? | Hoch → keine echten H übersehen |
| **F1-Score** | Balance aus beidem? | Hoch → gutes Gleichgewicht |
| **Macro-F1** | Gut für alle 3 Klassen? | Hoch → auch Unentschieden erkannt |
| **Weighted-F1** | Gesamtleistung gewichtet? | Hoch → gute Gesamtperformance |

> **Wichtiger Kontext:** Unentschieden (D) ist die schwierigste Klasse –
> sie kommt seltener vor und ist am wenigsten vorhersehbar. Ein hoher
> **Macro-F1** zeigt, dass das Modell auch Unentschieden erkennt – das ist
> die eigentliche Herausforderung dieser Aufgabe.
