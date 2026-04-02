"""
predict_spiel.py
================
Interaktives Vorhersage-Tool für Bundesliga-Spiele.

Lädt automatisch das beste Modell aus xx_Notebook (gespeichert vom Notebook).
Gibt Heim- und Auswärtsteam ein → das beste trainierte Modell liefert die Prognose.

Verwendung:
    python predict_spiel.py
    python predict_spiel.py --heim "Bayern Munich" --auswaerts "Dortmund"
"""

import argparse
import difflib
import glob
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional

from utils import (
    load_bundesliga_data, load_xg_data,
    compute_rolling_features, merge_xg_features,
    add_derived_features, normalize_team,
)

warnings.filterwarnings("ignore")

LABEL_TEXT = {"H": "Heimsieg ⚽", "D": "Unentschieden 🤝", "A": "Auswärtssieg ✈️"}

# ─────────────────────────────────────────────────────────────────────────────
# BESTES MODELL LADEN
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR  = os.path.join(BASE_DIR, "xx_Notebook")


def lade_bestes_modell() -> dict:
    """
    Sucht im xx_Notebook Ordner nach allen best_model_*.pkl Dateien
    und lädt die zuletzt gespeicherte (= aktuell beste).
    """
    pattern = os.path.join(NOTEBOOK_DIR, "best_model_*.pkl")
    dateien = glob.glob(pattern)

    if not dateien:
        raise FileNotFoundError(
            f"\n❌ Kein gespeichertes Modell gefunden in: {NOTEBOOK_DIR}\n"
            f"   Bitte zuerst das Notebook ausführen (Abschnitt 5.3)."
        )

    # Neueste Datei nehmen (nach Änderungsdatum)
    neueste = max(dateien, key=os.path.getmtime)
    package = joblib.load(neueste)

    print(f"✅ Modell geladen: {os.path.basename(neueste)}")
    print(f"   Typ:       {package['model_type']}")
    print(f"   Variante:  {package['variant']}")
    print(f"   F1 macro:  {package['metrics']['f1_macro']:.4f}")
    print(f"   F1 Draw:   {package['metrics']['f1_draw']:.4f}")
    print(f"   Accuracy:  {package['metrics']['accuracy']:.4f}")
    print(f"   Gespeichert am: {package['saved_at']}\n")

    return package


# ─────────────────────────────────────────────────────────────────────────────
# TEAM-SUCHE: Fuzzy Matching
# ─────────────────────────────────────────────────────────────────────────────
def find_team(eingabe: str, alle_teams: list) -> Optional[str]:
    norm = normalize_team(eingabe)
    if norm in alle_teams:
        return norm
    lower_map = {t.lower(): t for t in alle_teams}
    if eingabe.lower() in lower_map:
        return lower_map[eingabe.lower()]
    if norm.lower() in lower_map:
        return lower_map[norm.lower()]
    matches = difflib.get_close_matches(norm, alle_teams, n=3, cutoff=0.4)
    if matches:
        return matches[0]
    for team in alle_teams:
        if eingabe.lower() in team.lower() or team.lower() in eingabe.lower():
            return team
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES FÜR KOMMENDES SPIEL BERECHNEN
# ─────────────────────────────────────────────────────────────────────────────
def compute_upcoming_match_features(
    home_team: str,
    away_team: str,
    raw_df: pd.DataFrame,
    xg_df: pd.DataFrame,
    aktuelle_saison: str,
) -> tuple:
    """
    Berechnet die Rolling/EWA-Features für ein noch nicht gespieltes Spiel,
    indem eine Dummy-Zeile ans Ende der historischen Daten gehängt wird.
    Gibt (dummy_feat_ohne_xg, dummy_feat_mit_xg) zurück.
    """
    dummy_row = pd.DataFrame([{
        "Date":    pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
        "Season":  aktuelle_saison,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": np.nan, "FTAG": np.nan, "FTR": np.nan,
        "HS":   np.nan, "AS":   np.nan,
        "HST":  np.nan, "AST":  np.nan,
    }])

    df_combined  = pd.concat([raw_df, dummy_row], ignore_index=True)
    df_feat      = compute_rolling_features(df_combined)
    dummy_feat   = df_feat.iloc[-1:]

    dummy_feat_xg = dummy_feat.copy()
    dummy_feat_xg = merge_xg_features(dummy_feat_xg, xg_df)

    dummy_feat    = add_derived_features(dummy_feat,    include_xg=False)
    dummy_feat_xg = add_derived_features(dummy_feat_xg, include_xg=True)

    return dummy_feat.fillna(0.0), dummy_feat_xg.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# VORHERSAGE BERECHNEN & AUSGEBEN
# ─────────────────────────────────────────────────────────────────────────────
def drucke_vorhersage(
    home_team:     str,
    away_team:     str,
    dummy_feat:    pd.DataFrame,
    dummy_feat_xg: pd.DataFrame,
    package:       dict,
    xg_latest:     pd.DataFrame,
    xg_df:         pd.DataFrame,
) -> None:
    model      = package['model']
    scaler     = package['scaler']
    features   = package['features']
    le_classes = package['le_classes']        # ['A', 'D', 'H']
    model_type = package['model_type']
    variant    = package['variant']

    bar = "─" * 65

    # ── Feature-Vektor auswählen (mit oder ohne xG je nach Variante) ──────────
    # Die Variante aus dem Notebook verrät uns welche Features genutzt wurden
    use_xg = "xG" in variant or "Mit xG" in variant
    source_df = dummy_feat_xg if use_xg else dummy_feat

    # Sicherstellen dass alle Features vorhanden sind
    fehlende = [f for f in features if f not in source_df.columns]
    if fehlende:
        print(f"⚠️  Fehlende Features: {fehlende} — werden mit 0 aufgefüllt.")
        for f in fehlende:
            source_df[f] = 0.0

    X_neu = source_df[features].values

    # ── Skalieren & Vorhersagen ───────────────────────────────────────────────
    X_scaled = scaler.transform(X_neu)
    proba    = model.predict_proba(X_scaled)[0]

    # Wahrscheinlichkeiten den richtigen Klassen zuordnen
    if hasattr(model, 'classes_'):
        model_classes = list(model.classes_)
    else:
        model_classes = list(range(len(proba)))

    # Klassen-Index → Label ('A', 'D', 'H')
    if isinstance(model_classes[0], (int, np.integer)):
        # Encoded (0,1,2) → le_classes gibt die Reihenfolge
        p = {le_classes[i]: proba[i] for i in range(len(proba))}
    else:
        p = {str(c): proba[i] for i, c in enumerate(model_classes)}

    tip  = max(p, key=p.get)
    conf = p[tip]

    # ── Formwerte aus Features ────────────────────────────────────────────────
    def safe_get(df, col):
        return df[col].iloc[0] if col in df.columns else np.nan

    h_erz = safe_get(dummy_feat, "home_team_overall_ewa_GoalsScored")
    a_erz = safe_get(dummy_feat, "away_team_overall_ewa_GoalsScored")
    h_kas = safe_get(dummy_feat, "home_team_overall_ewa_GoalsConceded")
    a_kas = safe_get(dummy_feat, "away_team_overall_ewa_GoalsConceded")
    h_pts = safe_get(dummy_feat, "home_team_overall_ewa_Points")
    a_pts = safe_get(dummy_feat, "away_team_overall_ewa_Points")

    # ── xG-Daten ──────────────────────────────────────────────────────────────
    def get_xg(team):
        r = xg_latest[xg_latest["Team"] == team] if not xg_latest.empty else pd.DataFrame()
        if r.empty:
            r = xg_df[xg_df["Team"] == team].tail(1)
        if r.empty:
            return None
        v = r.iloc[0]
        return {
            "xG_per_game":  v.get("xG_per_game",  np.nan),
            "xGA_per_game": v.get("xGA_per_game", np.nan),
            "xPTS_per_game": v.get("xPTS_per_game", np.nan),
        }

    home_xg = get_xg(home_team)
    away_xg = get_xg(away_team)

    # ── Ausgabe ───────────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  ⚽  VORHERSAGE:  {home_team}  vs.  {away_team}")
    print(f"{bar}")
    print(f"  Modell:   {model_type} | Variante: {variant}")
    print(f"  Metriken: F1 macro={package['metrics']['f1_macro']:.3f} | "
          f"F1 Draw={package['metrics']['f1_draw']:.3f} | "
          f"Acc={package['metrics']['accuracy']:.3f}")

    print(f"\n📊 FORM (Exponentially Weighted Averages):")
    print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
    print(f"   {'Team':30} {home_team[-12:]:>12} {away_team[-12:]:>12}")
    print(f"   {'Gew. Ø Tore erzielt':30} {h_erz:>12.2f} {a_erz:>12.2f}")
    print(f"   {'Gew. Ø Tore kassiert':30} {h_kas:>12.2f} {a_kas:>12.2f}")
    print(f"   {'Gew. Ø Punkte/Spiel':30} {h_pts:>12.2f} {a_pts:>12.2f}")

    if home_xg and away_xg:
        print(f"\n📈 xG-DATEN (Vorjahreswerte als Prior):")
        print(f"   {'':30} {'HEIM':>12} {'AUSWÄRTS':>12}")
        print(f"   {'xG / Spiel':30} {home_xg.get('xG_per_game', np.nan):>12.2f} "
              f"{away_xg.get('xG_per_game', np.nan):>12.2f}")
        print(f"   {'xGA / Spiel':30} {home_xg.get('xGA_per_game', np.nan):>12.2f} "
              f"{away_xg.get('xGA_per_game', np.nan):>12.2f}")
        print(f"   {'xPTS / Spiel':30} {home_xg.get('xPTS_per_game', np.nan):>12.2f} "
              f"{away_xg.get('xPTS_per_game', np.nan):>12.2f}")

    # Grobe Torschätzung
    est_home = (h_erz + a_kas) / 2
    est_away = (a_erz + h_kas) / 2
    print(f"\n🎯 ERWARTETE TORE (grobe Schätzung):")
    print(f"   {home_team[-22:]:<24} {est_home:.2f}  |  "
          f"{away_team[-22:]:<24} {est_away:.2f}")

    # ── Hauptergebnis ─────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  📋  MODELL-VORHERSAGE ({model_type})")
    print(f"{bar}")
    print(f"  {'Ergebnis':<20} {'P(Heimsieg)':>13} {'P(Unentsch.)':>13} {'P(Auswärts)':>13}")
    print(f"  {'─'*60}")
    icon = "🏠" if tip == "H" else ("🤝" if tip == "D" else "✈️")
    print(f"  {icon} {LABEL_TEXT[tip]:<18} "
          f"{p.get('H', 0):>13.1%} {p.get('D', 0):>13.1%} {p.get('A', 0):>13.1%}")

    print(f"\n  🏆 TIPP:  {LABEL_TEXT[tip]}  (Konfidenz: {conf:.1%})")

    # Konfidenz-Warnung bei knappen Entscheidungen
    second = sorted(p.values(), reverse=True)[1]
    if conf - second < 0.05:
        print(f"  ⚠️  Knappe Entscheidung — Differenz zur 2. Klasse: {conf - second:.1%}")

    print(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bundesliga Spielvorhersage mit bestem Notebook-Modell"
    )
    parser.add_argument("--heim",       type=str, help="Heimteam")
    parser.add_argument("--auswaerts",  type=str, help="Auswärtsteam")
    parser.add_argument("--saison",     type=str, help="Saison (z.B. 2024/25)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  BUNDESLIGA VORHERSAGE — BESTES NOTEBOOK-MODELL")
    print("=" * 65 + "\n")

    # ── Bestes Modell laden ───────────────────────────────────────────────────
    package = lade_bestes_modell()

    # ── Rohdaten laden ────────────────────────────────────────────────────────
    print("📂 Lade Spieldaten ...")
    raw_df     = load_bundesliga_data()
    alle_teams = sorted(set(raw_df["HomeTeam"]) | set(raw_df["AwayTeam"]))
    saison     = args.saison or raw_df["Season"].max()

    xg_df = load_xg_data()

    # xG Vorjahreswerte als Prior
    saisons_sorted = sorted(xg_df["Season"].unique()) if not xg_df.empty else []
    if saison in saisons_sorted:
        idx          = saisons_sorted.index(saison)
        prior_saison = saisons_sorted[idx - 1] if idx > 0 else saison
    else:
        prior_saison = saisons_sorted[-1] if saisons_sorted else saison

    xg_latest = (xg_df[xg_df["Season"] == prior_saison]
                 if not xg_df.empty else pd.DataFrame())

    print(f"📅 Saison:    {saison}")
    print(f"📊 xG-Prior:  {prior_saison}")
    print(f"📋 Teams:     {len(alle_teams)} verfügbar\n")

    # ── Vorhersage-Loop ───────────────────────────────────────────────────────
    while True:
        heim_in = args.heim or input("🏠 Heimteam ('exit' zum Beenden): ").strip()
        if heim_in.lower() in ("exit", "q"):
            break
        args.heim = None

        home = find_team(heim_in, alle_teams)
        if not home:
            print(f"  ❌ '{heim_in}' nicht gefunden. Verfügbare Teams:\n"
                  f"  {', '.join(alle_teams[:10])} ...\n")
            continue
        if home != heim_in:
            print(f"  🔍 Gefunden: {home}")

        ausw_in = args.auswaerts or input("✈️  Auswärtsteam: ").strip()
        args.auswaerts = None

        away = find_team(ausw_in, alle_teams)
        if not away:
            print(f"  ❌ '{ausw_in}' nicht gefunden.\n")
            continue
        if away != ausw_in:
            print(f"  🔍 Gefunden: {away}")
        if home == away:
            print("  ❌ Heim- und Auswärtsteam dürfen nicht identisch sein.\n")
            continue

        print(f"\n  🔄 Berechne EWA-Features für {home} vs. {away} ...")
        dummy_feat, dummy_feat_xg = compute_upcoming_match_features(
            home, away, raw_df, xg_df, saison
        )

        drucke_vorhersage(
            home, away,
            dummy_feat, dummy_feat_xg,
            package,
            xg_latest, xg_df,
        )

        weiter = input("  ➡️  Noch ein Spiel? (Enter = ja / 'exit' = nein): ").strip().lower()
        if weiter in ("exit", "q"):
            break

    print("\n👋 Auf Wiedersehen!\n")


if __name__ == "__main__":
    main()