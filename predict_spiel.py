"""
predict_spiel.py
================
Interaktives Vorhersage-Tool für Bundesliga-Spiele.
Gibt Heim- und Auswärtsteam ein → alle 4 Modelle liefern ihre Prognose.

Vorgehensweise:
  - Alle verfügbaren Daten (alle Saisons) werden zum Training genutzt
  - Die Rolling-Features der letzten 20 Spiele pro Team werden verwendet
  - Ausgabe: Wahrscheinlichkeiten (H/D/A) + erwartete Tore aller 4 Modelle

Aufruf:
  python3 predict_spiel.py
  python3 predict_spiel.py --heim "Bayern Munich" --auswaerts "Dortmund"
"""

import argparse
import difflib
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from typing import Optional, Union

from utils import (
    prepare_dataset, BASE_FEATURES, XG_FEATURES,
    load_bundesliga_data, load_xg_data,
    compute_rolling_features, merge_xg_features,
    add_derived_features, normalize_team, ROLLING_N,
)

warnings.filterwarnings("ignore")

LABEL_ORDER = ["H", "D", "A"]
LABEL_TEXT  = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}


# ─────────────────────────────────────────────────────────────────────────────
# TEAM-SUCHE: Fuzzy Matching für Eingabe-Fehler
# ─────────────────────────────────────────────────────────────────────────────

def find_team(eingabe: str, alle_teams: list) -> Optional[str]:
    """Sucht den nächsten passenden Teamnamen (case-insensitive + fuzzy)."""
    # 1. Exaktes Match nach Normalisierung
    norm = normalize_team(eingabe)
    if norm in alle_teams:
        return norm

    # 2. Case-insensitive Suche in normalisierten Namen
    lower_map = {t.lower(): t for t in alle_teams}
    if eingabe.lower() in lower_map:
        return lower_map[eingabe.lower()]
    if norm.lower() in lower_map:
        return lower_map[norm.lower()]

    # 3. Fuzzy-Matching (difflib)
    matches = difflib.get_close_matches(norm, alle_teams, n=3, cutoff=0.4)
    if matches:
        return matches[0]

    # 4. Substring-Suche
    for team in alle_teams:
        if eingabe.lower() in team.lower() or team.lower() in eingabe.lower():
            return team

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODELLE TRAINIEREN (auf ALLEN Daten)
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models():
    """
    Trainiert alle 4 Modelle auf dem gesamten verfügbaren Datensatz.
    Gibt zurück: (models_ohne_xg, models_mit_xg, df_ohne, df_mit, features_ohne, features_mit)
    """
    print("📂 Lade & trainiere alle Modelle auf allen verfügbaren Daten ...")

    # Ohne xG
    df_ohne, feat_ohne = prepare_dataset(include_xg=False)
    X_ohne = df_ohne[feat_ohne].values
    y_ohne = df_ohne["FTR"].values

    # Mit xG
    df_mit, feat_mit = prepare_dataset(include_xg=True)
    X_mit = df_mit[feat_mit].values
    y_mit = df_mit["FTR"].values

    def make_models():
        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    multi_class="multinomial", solver="lbfgs",
                    max_iter=1000, C=1.0, random_state=42
                )),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                )),
            ]),
        }

    models_ohne = make_models()
    for name, m in models_ohne.items():
        print(f"  🔧 Trainiere (OHNE xG): {name} ...")
        m.fit(X_ohne, y_ohne)

    models_mit = make_models()
    for name, m in models_mit.items():
        print(f"  🔧 Trainiere (MIT xG):  {name} ...")
        m.fit(X_mit, y_mit)

    print("✅ Alle 4 Modelle trainiert.\n")
    return models_ohne, models_mit, df_ohne, df_mit, feat_ohne, feat_mit


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING FEATURES FÜR EIN EINZELNES SPIEL BERECHNEN
# ─────────────────────────────────────────────────────────────────────────────

def get_team_rolling_stats(team: str, df: pd.DataFrame, n: int = ROLLING_N) -> dict:
    """
    Gibt die Rolling-Stats (letzte n Spiele) eines Teams aus dem DataFrame zurück.
    """
    # Alle Spiele des Teams (Heim oder Auswärts)
    heim    = df[df["HomeTeam"] == team].copy()
    ausw    = df[df["AwayTeam"] == team].copy()

    heim = heim.assign(
        goals_scored   = heim["FTHG"],
        goals_conceded = heim["FTAG"],
        shots          = heim.get("HS",  pd.Series(dtype=float)),
        sot            = heim.get("HST", pd.Series(dtype=float)),
        points         = heim["FTR"].map({"H": 3, "D": 1, "A": 0})
    )
    ausw = ausw.assign(
        goals_scored   = ausw["FTAG"],
        goals_conceded = ausw["FTHG"],
        shots          = ausw.get("AS",  pd.Series(dtype=float)),
        sot            = ausw.get("AST", pd.Series(dtype=float)),
        points         = ausw["FTR"].map({"A": 3, "D": 1, "H": 0})
    )

    alle = pd.concat([
        heim[["Date", "goals_scored", "goals_conceded", "shots", "sot", "points"]],
        ausw[["Date", "goals_scored", "goals_conceded", "shots", "sot", "points"]],
    ]).sort_values("Date").tail(n)

    if len(alle) == 0:
        return {}

    stats = {
        "avg_goals_scored":    alle["goals_scored"].mean(),
        "avg_goals_conceded":  alle["goals_conceded"].mean(),
        "avg_points":          alle["points"].mean(),
        "win_rate":            (alle["points"] == 3).mean(),
        "draw_rate":           (alle["points"] == 1).mean(),
        "loss_rate":           (alle["points"] == 0).mean(),
        "avg_shots":           alle["shots"].mean() if "shots" in alle else np.nan,
        "avg_shots_on_target": alle["sot"].mean()   if "sot"   in alle else np.nan,
        "spiele":              len(alle),
    }
    return stats


def build_feature_vector(home_stats: dict, away_stats: dict,
                          features: list,
                          home_xg: Optional[dict] = None,
                          away_xg: Optional[dict] = None) -> Optional[np.ndarray]:
    """Baut den Feature-Vektor für ein einzelnes Spiel aus Rolling-Stats."""
    mapping = {
        "home_avg_goals_scored":    home_stats.get("avg_goals_scored", np.nan),
        "home_avg_goals_conceded":  home_stats.get("avg_goals_conceded", np.nan),
        "home_avg_points":          home_stats.get("avg_points", np.nan),
        "home_win_rate":            home_stats.get("win_rate", np.nan),
        "home_draw_rate":           home_stats.get("draw_rate", np.nan),
        "home_loss_rate":           home_stats.get("loss_rate", np.nan),
        "home_avg_shots":           home_stats.get("avg_shots", np.nan),
        "home_avg_shots_on_target": home_stats.get("avg_shots_on_target", np.nan),
        "away_avg_goals_scored":    away_stats.get("avg_goals_scored", np.nan),
        "away_avg_goals_conceded":  away_stats.get("avg_goals_conceded", np.nan),
        "away_avg_points":          away_stats.get("avg_points", np.nan),
        "away_win_rate":            away_stats.get("win_rate", np.nan),
        "away_draw_rate":           away_stats.get("draw_rate", np.nan),
        "away_loss_rate":           away_stats.get("loss_rate", np.nan),
        "away_avg_shots":           away_stats.get("avg_shots", np.nan),
        "away_avg_shots_on_target": away_stats.get("avg_shots_on_target", np.nan),
        "goal_diff_avg":            home_stats.get("avg_goals_scored", np.nan) - away_stats.get("avg_goals_scored", np.nan),
        "points_diff_avg":          home_stats.get("avg_points", np.nan)       - away_stats.get("avg_points", np.nan),
    }

    if home_xg and away_xg:
        mapping.update({
            "home_xG_per_game":   home_xg.get("xG_per_game",   np.nan),
            "home_xGA_per_game":  home_xg.get("xGA_per_game",  np.nan),
            "home_xPTS_per_game": home_xg.get("xPTS_per_game", np.nan),
            "away_xG_per_game":   away_xg.get("xG_per_game",   np.nan),
            "away_xGA_per_game":  away_xg.get("xGA_per_game",  np.nan),
            "away_xPTS_per_game": away_xg.get("xPTS_per_game", np.nan),
            "xG_diff":  home_xg.get("xG_per_game",  np.nan) - away_xg.get("xG_per_game",  np.nan),
            "xGA_diff": home_xg.get("xGA_per_game", np.nan) - away_xg.get("xGA_per_game", np.nan),
        })

    vec = np.array([mapping.get(f, np.nan) for f in features])

    if np.any(np.isnan(vec)):
        # Fehlende Werte mit Median füllen (als Fallback)
        vec = np.where(np.isnan(vec), 0.0, vec)

    return vec.reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# AUSGABE
# ─────────────────────────────────────────────────────────────────────────────

def drucke_vorhersage(home_team: str, away_team: str,
                       home_stats: dict, away_stats: dict,
                       models_ohne: dict, models_mit: dict,
                       feat_ohne: list, feat_mit: list,
                       home_xg: Optional[dict], away_xg: Optional[dict]):
    """Druckt die vollständige Vorhersage für ein Spiel."""

    bar = "─" * 65
    print(f"\n{bar}")
    print(f"  ⚽  VORHERSAGE: {home_team}  vs.  {away_team}")
    print(f"{bar}")

    # ── Form-Übersicht ──────────────────────────────────────────────
    print(f"\n📊 FORM (letzte {home_stats.get('spiele', '?')} / {away_stats.get('spiele', '?')} Spiele):")
    print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
    print(f"   {'Team':30} {home_team[-12:]:>12} {away_team[-12:]:>12}")
    print(f"   {'Ø Tore erzielt':30} {home_stats.get('avg_goals_scored', 0):>12.2f} {away_stats.get('avg_goals_scored', 0):>12.2f}")
    print(f"   {'Ø Tore kassiert':30} {home_stats.get('avg_goals_conceded', 0):>12.2f} {away_stats.get('avg_goals_conceded', 0):>12.2f}")
    print(f"   {'Ø Punkte/Spiel':30} {home_stats.get('avg_points', 0):>12.2f} {away_stats.get('avg_points', 0):>12.2f}")
    print(f"   {'Siegquote':30} {home_stats.get('win_rate', 0):>12.1%} {away_stats.get('win_rate', 0):>12.1%}")
    print(f"   {'Unentschieden-Quote':30} {home_stats.get('draw_rate', 0):>12.1%} {away_stats.get('draw_rate', 0):>12.1%}")
    print(f"   {'Niederlage-Quote':30} {home_stats.get('loss_rate', 0):>12.1%} {away_stats.get('loss_rate', 0):>12.1%}")

    if home_xg and away_xg:
        print(f"\n📈 xG-SAISONDATEN:")
        print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
        print(f"   {'xG / Spiel':30} {home_xg.get('xG_per_game', float('nan')):>12.2f} {away_xg.get('xG_per_game', float('nan')):>12.2f}")
        print(f"   {'xGA / Spiel':30} {home_xg.get('xGA_per_game', float('nan')):>12.2f} {away_xg.get('xGA_per_game', float('nan')):>12.2f}")
        print(f"   {'xPTS / Spiel':30} {home_xg.get('xPTS_per_game', float('nan')):>12.2f} {away_xg.get('xPTS_per_game', float('nan')):>12.2f}")

    # ── Erwartete Tore (einfache Schätzung aus Rolling-Stats) ────────
    est_home_goals = (home_stats.get("avg_goals_scored", 1.5)  + away_stats.get("avg_goals_conceded", 1.5)) / 2
    est_away_goals = (away_stats.get("avg_goals_scored", 1.2)  + home_stats.get("avg_goals_conceded", 1.2)) / 2

    if home_xg and away_xg and not np.isnan(home_xg.get("xG_per_game", np.nan)):
        est_home_goals_xg = (home_xg["xG_per_game"]  + away_xg["xGA_per_game"]) / 2
        est_away_goals_xg = (away_xg["xG_per_game"]  + home_xg["xGA_per_game"]) / 2
    else:
        est_home_goals_xg = None
        est_away_goals_xg = None

    print(f"\n🎯 ERWARTETE TORE (Schätzung):")
    print(f"   OHNE xG:  {home_team[-20:]:<22} {est_home_goals:.2f}  |  {away_team[-20:]:<22} {est_away_goals:.2f}")
    if est_home_goals_xg is not None:
        print(f"   MIT xG:   {home_team[-20:]:<22} {est_home_goals_xg:.2f}  |  {away_team[-20:]:<22} {est_away_goals_xg:.2f}")
    print()

    # ── Modell-Vorhersagen ────────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"  📋  MODELL-VORHERSAGEN")
    print(f"{'─'*65}")
    print(f"  {'Modell':<30} {'Tip':>10} {'P(Heim)':>9} {'P(Unt.)':>9} {'P(Ausw.)':>9}")
    print(f"  {'─'*60}")

    vec_ohne = build_feature_vector(home_stats, away_stats, feat_ohne)
    vec_mit  = build_feature_vector(home_stats, away_stats, feat_mit, home_xg, away_xg)

    for variant, models, vec, label in [
        ("OHNE xG", models_ohne, vec_ohne, ""),
        ("MIT xG",  models_mit,  vec_mit,  ""),
    ]:
        for name, model in models.items():
            proba = model.predict_proba(vec)[0]
            # proba ist in der Reihenfolge der trainierten Klassen
            classes = list(model.classes_)
            p = {c: proba[i] for i, c in enumerate(classes)}
            p_h = p.get("H", 0)
            p_d = p.get("D", 0)
            p_a = p.get("A", 0)
            tip = max(p, key=p.get)
            tip_text = LABEL_TEXT[tip]
            model_label = f"{name} [{variant}]"
            tip_icon = "🏠" if tip == "H" else ("🤝" if tip == "D" else "✈️")
            print(f"  {model_label:<30} {tip_icon} {tip_text:<11} {p_h:>8.1%} {p_d:>9.1%} {p_a:>9.1%}")

    print(f"{'─'*65}")

    # ── Konsens ───────────────────────────────────────────────────────
    all_tips = []
    for models, vec in [(models_ohne, vec_ohne), (models_mit, vec_mit)]:
        for model in models.values():
            classes = list(model.classes_)
            proba = model.predict_proba(vec)[0]
            p = {c: proba[i] for i, c in enumerate(classes)}
            all_tips.append(max(p, key=p.get))

    from collections import Counter
    c = Counter(all_tips)
    majority = c.most_common(1)[0][0]
    count = c.most_common(1)[0][1]
    tip_icon = "🏠" if majority == "H" else ("🤝" if majority == "D" else "✈️")
    print(f"\n  🗳️  KONSENS ({count}/4 Modelle): {tip_icon} {LABEL_TEXT[majority]}")
    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bundesliga Spielvorhersage")
    parser.add_argument("--heim",      type=str, default=None, help="Heimteam")
    parser.add_argument("--auswaerts", type=str, default=None, help="Auswärtsteam")
    parser.add_argument("--saison",    type=str, default=None, help="Aktuelle Saison für xG-Daten (z.B. '2025/26')")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  BUNDESLIGA LIVE-VORHERSAGE")
    print("=" * 65)

    # 1. Modelle trainieren
    models_ohne, models_mit, df_ohne, df_mit, feat_ohne, feat_mit = train_all_models()

    # Alle bekannten Teams
    alle_teams = sorted(set(df_ohne["HomeTeam"].tolist() + df_ohne["AwayTeam"].tolist()))
    aktuelle_saison = args.saison or df_ohne["Season"].max()

    # xG-Daten laden
    xg_df = load_xg_data()
    xg_latest = xg_df[xg_df["Season"] == aktuelle_saison] if aktuelle_saison in xg_df["Season"].values else xg_df[xg_df["Season"] == xg_df["Season"].max()]

    print(f"\n📅 Aktuelle Saison für xG-Daten: {aktuelle_saison}")
    print(f"📋 {len(alle_teams)} Teams verfügbar\n")
    print("   Tipp: Du kannst auch Kurzformen eingeben, z.B. 'Bayern', 'BVB', 'Gladbach'\n")

    # Dauerschleife für mehrere Spiele
    while True:
        print("─" * 65)

        # Heimteam
        if args.heim:
            heim_eingabe = args.heim
            args.heim = None  # nur einmal aus args
        else:
            heim_eingabe = input("🏠 Heimteam (oder 'exit' zum Beenden): ").strip()

        if heim_eingabe.lower() in ("exit", "quit", "q", "beenden"):
            print("\n👋 Bis dann!\n")
            break

        home_team = find_team(heim_eingabe, alle_teams)
        if not home_team:
            print(f"  ❌ Team '{heim_eingabe}' nicht gefunden.")
            print(f"  Verfügbare Teams: {', '.join(alle_teams)}")
            continue
        if home_team != heim_eingabe:
            print(f"  → Erkannt als: {home_team}")

        # Auswärtsteam
        if args.auswaerts:
            ausw_eingabe = args.auswaerts
            args.auswaerts = None
        else:
            ausw_eingabe = input("✈️  Auswärtsteam: ").strip()

        away_team = find_team(ausw_eingabe, alle_teams)
        if not away_team:
            print(f"  ❌ Team '{ausw_eingabe}' nicht gefunden.")
            continue
        if away_team != ausw_eingabe:
            print(f"  → Erkannt als: {away_team}")

        if home_team == away_team:
            print("  ❌ Heim- und Auswärtsteam müssen unterschiedlich sein.")
            continue

        # 2. Rolling Stats berechnen (aus allen vorhandenen Daten)
        # Wir brauchen das rohe Bundesliga-DF (mit FTHG/FTAG/HS etc.)
        raw_df = load_bundesliga_data()

        home_stats = get_team_rolling_stats(home_team, raw_df, n=ROLLING_N)
        away_stats = get_team_rolling_stats(away_team, raw_df, n=ROLLING_N)

        if not home_stats:
            print(f"  ⚠️  Keine Spieldaten für '{home_team}' gefunden!")
            continue
        if not away_stats:
            print(f"  ⚠️  Keine Spieldaten für '{away_team}' gefunden!")
            continue

        # 3. xG-Daten für aktuelle Saison
        def get_xg(team):
            row = xg_latest[xg_latest["Team"] == team]
            if row.empty:
                # Fallback: letzte Saison mit Daten
                row = xg_df[xg_df["Team"] == team].sort_values("Season").tail(1)
            if row.empty:
                return None
            r = row.iloc[0]
            return {
                "xG_per_game":   r.get("xG_per_game",   np.nan),
                "xGA_per_game":  r.get("xGA_per_game",  np.nan),
                "xPTS_per_game": r.get("xPTS_per_game", np.nan),
            }

        home_xg = get_xg(home_team)
        away_xg = get_xg(away_team)

        if not home_xg or not away_xg:
            print(f"  ⚠️  Keine xG-Daten für diese Teams – xG-Modell nutzt Fallback-Nullwerte")

        # 4. Vorhersage ausgeben
        drucke_vorhersage(
            home_team, away_team,
            home_stats, away_stats,
            models_ohne, models_mit,
            feat_ohne, feat_mit,
            home_xg, away_xg,
        )

        # Nächstes Spiel?
        weiter = input("  ➡️  Noch ein Spiel vorhersagen? (Enter = ja / 'exit' = nein): ").strip()
        if weiter.lower() in ("exit", "q", "nein", "n", "no"):
            print("\n👋 Bis dann!\n")
            break


if __name__ == "__main__":
    main()
