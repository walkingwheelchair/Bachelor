"""
predict_spiel.py
================
Interaktives Vorhersage-Tool für Bundesliga-Spiele.
Gibt Heim- und Auswärtsteam ein → alle Modelle liefern ihre Prognose.
Nutzt die neuen Exponentially Weighted Averages (EWA) und XGBoost.
"""

import argparse
import difflib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from typing import Optional

from utils import (
    prepare_dataset, BASE_FEATURES, XG_FEATURES,
    load_bundesliga_data, load_xg_data,
    compute_rolling_features, merge_xg_features,
    add_derived_features, normalize_team
)

warnings.filterwarnings("ignore")

LABEL_ORDER = ["H", "D", "A"]
LABEL_TEXT  = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}

# ─────────────────────────────────────────────────────────────────────────────
# TEAM-SUCHE: Fuzzy Matching für Eingabe-Fehler
# ─────────────────────────────────────────────────────────────────────────────
def find_team(eingabe: str, alle_teams: list) -> Optional[str]:
    norm = normalize_team(eingabe)
    if norm in alle_teams: return norm
    lower_map = {t.lower(): t for t in alle_teams}
    if eingabe.lower() in lower_map: return lower_map[eingabe.lower()]
    if norm.lower() in lower_map: return lower_map[norm.lower()]
    matches = difflib.get_close_matches(norm, alle_teams, n=3, cutoff=0.4)
    if matches: return matches[0]
    for team in alle_teams:
        if eingabe.lower() in team.lower() or team.lower() in eingabe.lower():
            return team
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MODELLE TRAINIEREN
# ─────────────────────────────────────────────────────────────────────────────
def train_all_models():
    print("📂 Lade & trainiere alle Modelle auf allen verfügbaren Daten ...")
    df_ohne, feat_ohne = prepare_dataset(include_xg=False)
    X_ohne = df_ohne[feat_ohne].values
    y_ohne = df_ohne["FTR"].values

    df_mit, feat_mit = prepare_dataset(include_xg=True)
    X_mit = df_mit[feat_mit].values
    y_mit = df_mit["FTR"].values

    le = LabelEncoder()
    y_ohne_enc = le.fit_transform(y_ohne)
    y_mit_enc  = le.transform(y_mit)

    def make_models():
        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42)),
            ]),
            "XGBoost": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)),
            ]),
        }

    models_ohne = make_models()
    for name, m in models_ohne.items():
        print(f"  🔧 Trainiere (OHNE xG): {name} ...")
        m.fit(X_ohne, y_ohne_enc if "XGBoost" in name else y_ohne)

    models_mit = make_models()
    for name, m in models_mit.items():
        print(f"  🔧 Trainiere (MIT xG):  {name} ...")
        m.fit(X_mit, y_mit_enc if "XGBoost" in name else y_mit)

    print("✅ Alle 4 Modelle trainiert.\n")
    return models_ohne, models_mit, feat_ohne, feat_mit, le

# ─────────────────────────────────────────────────────────────────────────────
# DUMMY ROW EXTRACTION FÜR EWA
# ─────────────────────────────────────────────────────────────────────────────
def compute_upcoming_match_features(home_team: str, away_team: str, raw_df: pd.DataFrame, xg_df: pd.DataFrame, aktuelle_saison: str):
    dummy_row = pd.DataFrame([{
        "Date": pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        "Season": aktuelle_saison,
        "HomeTeam": home_team, "AwayTeam": away_team,
        "FTHG": np.nan, "FTAG": np.nan, "FTR": np.nan,
        "HS": np.nan, "AS": np.nan, "HST": np.nan, "AST": np.nan
    }])
    df_combined = pd.concat([raw_df, dummy_row], ignore_index=True)
    df_feat = compute_rolling_features(df_combined)
    dummy_feat = df_feat.iloc[-1:]
    
    dummy_feat_xg = dummy_feat.copy()
    dummy_feat_xg = merge_xg_features(dummy_feat_xg, xg_df)
    
    dummy_feat = add_derived_features(dummy_feat, include_xg=False)
    dummy_feat_xg = add_derived_features(dummy_feat_xg, include_xg=True)
    
    return dummy_feat.fillna(0.0), dummy_feat_xg.fillna(0.0)

# ─────────────────────────────────────────────────────────────────────────────
# AUSGABE
# ─────────────────────────────────────────────────────────────────────────────
def drucke_vorhersage(home_team, away_team, dummy_feat, dummy_feat_xg, models_ohne, models_mit, feat_ohne, feat_mit, le, home_xg, away_xg):
    bar = "─" * 65
    print(f"\n{bar}\n  ⚽  VORHERSAGE: {home_team}  vs.  {away_team}\n{bar}")

    h_erz = dummy_feat["home_team_overall_ewa_GoalsScored"].iloc[0]
    a_erz = dummy_feat["away_team_overall_ewa_GoalsScored"].iloc[0]
    h_kas = dummy_feat["home_team_overall_ewa_GoalsConceded"].iloc[0]
    a_kas = dummy_feat["away_team_overall_ewa_GoalsConceded"].iloc[0]
    h_pts = dummy_feat["home_team_overall_ewa_Points"].iloc[0]
    a_pts = dummy_feat["away_team_overall_ewa_Points"].iloc[0]

    print(f"\n📊 FORM (Exponentially Weighted Averages):")
    print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
    print(f"   {'Team':30} {home_team[-12:]:>12} {away_team[-12:]:>12}")
    print(f"   {'Gew. Ø Tore erzielt':30} {h_erz:>12.2f} {a_erz:>12.2f}")
    print(f"   {'Gew. Ø Tore kassiert':30} {h_kas:>12.2f} {a_kas:>12.2f}")
    print(f"   {'Gew. Ø Punkte/Spiel':30} {h_pts:>12.2f} {a_pts:>12.2f}")

    if home_xg and away_xg:
        print(f"\n📈 xG-SAISONDATEN:")
        print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
        print(f"   {'xG / Spiel':30} {home_xg.get('xG_per_game', np.nan):>12.2f} {away_xg.get('xG_per_game', np.nan):>12.2f}")
        print(f"   {'xGA / Spiel':30} {home_xg.get('xGA_per_game', np.nan):>12.2f} {away_xg.get('xGA_per_game', np.nan):>12.2f}")
        print(f"   {'xPTS / Spiel':30} {home_xg.get('xPTS_per_game', np.nan):>12.2f} {away_xg.get('xPTS_per_game', np.nan):>12.2f}")

    est_home_goals = (h_erz + a_kas) / 2
    est_away_goals = (a_erz + h_kas) / 2
    if home_xg and away_xg and not pd.isna(home_xg.get("xG_per_game")):
        est_h_xg = (home_xg["xG_per_game"] + away_xg["xGA_per_game"]) / 2
        est_a_xg = (away_xg["xG_per_game"] + home_xg["xGA_per_game"]) / 2
    else: est_h_xg = est_a_xg = None

    print(f"\n🎯 ERWARTETE TORE (Grobe Schätzung):")
    print(f"   OHNE xG:  {home_team[-20:]:<22} {est_home_goals:.2f}  |  {away_team[-20:]:<22} {est_away_goals:.2f}")
    if est_h_xg is not None:
        print(f"   MIT xG:   {home_team[-20:]:<22} {est_h_xg:.2f}  |  {away_team[-20:]:<22} {est_a_xg:.2f}")

    print(f"\n{bar}\n  📋  MODELL-VORHERSAGEN\n{bar}")
    print(f"  {'Modell':<30} {'Tip':>10} {'P(Heim)':>9} {'P(Unt.)':>9} {'P(Ausw.)':>9}\n  {'─'*60}")

    v_ohne = dummy_feat[feat_ohne].values
    v_mit  = dummy_feat_xg[feat_mit].values
    all_tips = []

    for variant, models, vec in [("OHNE xG", models_ohne, v_ohne), ("MIT xG", models_mit, v_mit)]:
        for name, model in models.items():
            proba = model.predict_proba(vec)[0]
            classes = list(le.inverse_transform(model.classes_)) if "XGBoost" in name else list(model.classes_)
            p = {c: proba[i] for i, c in enumerate(classes)}
            tip = max(p, key=p.get)
            all_tips.append(tip)
            icon = "🏠" if tip == "H" else ("🤝" if tip == "D" else "✈️")
            print(f"  {f'{name} [{variant}]':<30} {icon} {LABEL_TEXT[tip]:<11} {p.get('H',0):>8.1%} {p.get('D',0):>9.1%} {p.get('A',0):>9.1%}")

    from collections import Counter
    c = Counter(all_tips)
    maj = c.most_common(1)[0][0]
    print(f"{bar}\n  🗳️  KONSENS ({c.most_common(1)[0][1]}/4 Modelle): {'🏠' if maj == 'H' else ('🤝' if maj == 'D' else '✈️')} {LABEL_TEXT[maj]}\n{bar}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heim", type=str)
    parser.add_argument("--auswaerts", type=str)
    parser.add_argument("--saison", type=str)
    args = parser.parse_args()

    print("\n" + "=" * 65 + "\n  BUNDESLIGA LIVE-VORHERSAGE (MIT XGBOOST & EWA)\n" + "=" * 65)
    models_ohne, models_mit, feat_ohne, feat_mit, le = train_all_models()

    raw_df = load_bundesliga_data()
    alle_teams = sorted(set(raw_df["HomeTeam"]) | set(raw_df["AwayTeam"]))
    saison = args.saison or raw_df["Season"].max()

    xg_df = load_xg_data()
    xg_latest = xg_df[xg_df["Season"] == saison] if not xg_df.empty and saison in xg_df["Season"].values else (xg_df[xg_df["Season"] == xg_df["Season"].max()] if not xg_df.empty else pd.DataFrame())

    print(f"\n📅 Aktuelle Saison für xG-Daten: {saison}\n📋 {len(alle_teams)} Teams verfügbar\n")

    while True:
        heim_in = args.heim or input("🏠 Heimteam ('exit' zum Beenden): ").strip()
        if heim_in.lower() in ("exit", "q"): break
        args.heim = None

        home = find_team(heim_in, alle_teams)
        if not home:
            print("  ❌ Team nicht gefunden.")
            continue

        ausw_in = args.auswaerts or input("✈️  Auswärtsteam: ").strip()
        args.auswaerts = None
        away = find_team(ausw_in, alle_teams)
        if not away or home == away:
            print("  ❌ Team nicht gefunden oder identisch.")
            continue

        print(f"  🔄 Berechne aktuelle dynamische EWA-Form ...")
        dummy_feat, dummy_feat_xg = compute_upcoming_match_features(home, away, raw_df, xg_df, saison)

        def get_xg(team):
            r = xg_latest[xg_latest["Team"] == team]
            if r.empty: r = xg_df[xg_df["Team"] == team].tail(1)
            if r.empty: return None
            v = r.iloc[0]
            return {"xG_per_game": v.get("xG_per_game", np.nan), "xGA_per_game": v.get("xGA_per_game", np.nan), "xPTS_per_game": v.get("xPTS_per_game", np.nan)}

        home_xg, away_xg = get_xg(home), get_xg(away)
        drucke_vorhersage(home, away, dummy_feat, dummy_feat_xg, models_ohne, models_mit, feat_ohne, feat_mit, le, home_xg, away_xg)

        if input("  ➡️  Noch ein Spiel? (Enter/exit): ").strip().lower() in ("exit", "q"): break

if __name__ == "__main__":
    main()
