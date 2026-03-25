"""
utils.py
========
Gemeinsame Hilfsfunktionen für das Bundesliga-Vorhersageprojekt.
Enthält:
  - Team-Name-Mapping (Bundesliga CSV <-> xG CSV)
  - Datenladen & Zusammenführen aller Saisons
  - Rolling-Window-Feature-Berechnung (letzte N Spiele)
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PFADE
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDESLIGA_DIR = os.path.join(BASE_DIR, "Bundesliga_Quellen")
XG_DIR = os.path.join(BASE_DIR, "xG_Quellen")

# ─────────────────────────────────────────────────────────────────────────────
# TEAM-NAME-MAPPING
# Bundesliga-CSV-Namen → xG-CSV-Namen (normalisierte Form)
# ─────────────────────────────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    # Bayern
    "Bayern Munich":        "Bayern Munich",
    "FC Bayern Munich":     "Bayern Munich",

    # Dortmund
    "Dortmund":             "Borussia Dortmund",
    "Borussia Dortmund":    "Borussia Dortmund",

    # RB Leipzig
    "RB Leipzig":           "RasenBallsport Leipzig",
    "Leipzig":              "RasenBallsport Leipzig",
    "RasenBallsport Leipzig":"RasenBallsport Leipzig",

    # Hoffenheim
    "Hoffenheim":           "Hoffenheim",
    "TSG Hoffenheim":       "Hoffenheim",
    "TSG 1899 Hoffenheim":  "Hoffenheim",

    # Bayer Leverkusen
    "Leverkusen":           "Bayer Leverkusen",
    "Bayer Leverkusen":     "Bayer Leverkusen",

    # Borussia Mönchengladbach
    "M'gladbach":           "Borussia M.Gladbach",
    "Monchengladbach":      "Borussia M.Gladbach",
    "Borussia M.Gladbach":  "Borussia M.Gladbach",
    "B. Monchengladbach":   "Borussia M.Gladbach",
    "Mgladbach":            "Borussia M.Gladbach",

    # Eintracht Frankfurt
    "Ein Frankfurt":        "Eintracht Frankfurt",
    "Eintracht Frankfurt":  "Eintracht Frankfurt",
    "Frankfurt":            "Eintracht Frankfurt",

    # Schalke 04
    "Schalke 04":           "Schalke 04",
    "Schalke":              "Schalke 04",

    # Werder Bremen
    "Werder Bremen":        "Werder Bremen",
    "Bremen":               "Werder Bremen",

    # Hertha BSC
    "Hertha":               "Hertha Berlin",
    "Hertha Berlin":        "Hertha Berlin",
    "Hertha BSC":           "Hertha Berlin",

    # Augsburg
    "Augsburg":             "Augsburg",
    "FC Augsburg":          "Augsburg",

    # Hamburg
    "Hamburg":              "Hamburger SV",
    "Hamburger SV":         "Hamburger SV",

    # Freiburg
    "Freiburg":             "Freiburg",
    "SC Freiburg":          "Freiburg",

    # Mainz
    "Mainz":                "Mainz 05",
    "Mainz 05":             "Mainz 05",
    "FSV Mainz 05":         "Mainz 05",

    # Wolfsburg
    "Wolfsburg":            "Wolfsburg",
    "VfL Wolfsburg":        "Wolfsburg",

    # FC Köln
    "FC Koln":              "FC Cologne",
    "Koln":                 "FC Cologne",
    "FC Cologne":           "FC Cologne",
    "1. FC Köln":           "FC Cologne",

    # Darmstadt
    "Darmstadt":            "Darmstadt 98",
    "Darmstadt 98":         "Darmstadt 98",
    "SV Darmstadt 98":      "Darmstadt 98",

    # Ingolstadt
    "Ingolstadt":           "FC Ingolstadt 04",
    "FC Ingolstadt":        "FC Ingolstadt 04",
    "FC Ingolstadt 04":     "FC Ingolstadt 04",

    # Stuttgart (falls vorhanden)
    "Stuttgart":            "VfB Stuttgart",
    "VfB Stuttgart":        "VfB Stuttgart",

    # Hannover
    "Hannover":             "Hannover 96",
    "Hannover 96":          "Hannover 96",

    # Düsseldorf
    "Fortuna Dusseldorf":   "Fortuna Düsseldorf",
    "Dusseldorf":           "Fortuna Düsseldorf",
    "Fortuna Düsseldorf":   "Fortuna Düsseldorf",

    # Paderborn
    "Paderborn":            "SC Paderborn 07",
    "SC Paderborn":         "SC Paderborn 07",
    "SC Paderborn 07":      "SC Paderborn 07",

    # Union Berlin
    "Union Berlin":         "Union Berlin",
    "1. FC Union Berlin":   "Union Berlin",

    # Köln (nochmal)
    "1. FC Koln":           "FC Cologne",

    # Bochum
    "Bochum":               "VfL Bochum",
    "VfL Bochum":           "VfL Bochum",

    # Heidenheim
    "Heidenheim":           "1. FC Heidenheim 1846",
    "1. FC Heidenheim":     "1. FC Heidenheim 1846",
    "1. FC Heidenheim 1846": "1. FC Heidenheim 1846",

    # Holstein Kiel
    "Holstein Kiel":        "Holstein Kiel",

    # St. Pauli
    "St Pauli":             "FC St. Pauli",
    "FC St. Pauli":         "FC St. Pauli",

    # Nurnberg
    "Nurnberg":             "1. FC Nürnberg",
    "1. FC Nurnberg":       "1. FC Nürnberg",
    "1. FC Nürnberg":       "1. FC Nürnberg",
}


def normalize_team(name: str) -> str:
    """Normalisiert einen Teamnamen über das Mapping."""
    name = str(name).strip()
    return TEAM_NAME_MAP.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# DATENLADEN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_season_label(filename: str) -> str:
    """Extrahiert die Saisonbezeichnung aus dem Dateinamen, z.B. '2016:2017' -> '2016/17'."""
    base = os.path.basename(filename)
    # Format: "Bundesliga 2016:2017.csv" oder "xG 2016:2017.csv"
    parts = base.replace(".csv", "").split()
    season_raw = parts[-1]  # z.B. "2016:2017"
    years = season_raw.replace(":", "/").split("/")
    if len(years) == 2:
        return f"{years[0]}/{years[1][2:]}"  # -> "2016/17"
    return season_raw


def load_bundesliga_data() -> pd.DataFrame:
    """Lädt alle Bundesliga-Spieldaten und gibt einen DataFrame zurück."""
    files = sorted(glob.glob(os.path.join(BUNDESLIGA_DIR, "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(f, encoding="latin-1", on_bad_lines="skip")

        df["Season"] = _parse_season_label(f)

        # Nur relevante Spalten behalten
        cols_needed = ["Season", "Date", "HomeTeam", "AwayTeam",
                       "FTHG", "FTAG", "FTR",
                       "HS", "AS", "HST", "AST",
                       "HTHG", "HTAG"]
        available = [c for c in cols_needed if c in df.columns]
        df = df[available].dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

        # Teamnamen normalisieren
        df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
        df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)

        # Datum parsen
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.dropna(subset=["FTR"])
    all_data["FTR"] = all_data["FTR"].str.strip()
    all_data = all_data[all_data["FTR"].isin(["H", "D", "A"])]

    # Numerische Spalten sicherstellen
    for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HTHG", "HTAG"]:
        if col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    all_data = all_data.sort_values(["Season", "Date"]).reset_index(drop=True)
    return all_data


def load_xg_data() -> pd.DataFrame:
    """Lädt alle xG-Saisondaten und gibt einen DataFrame zurück."""
    files = sorted(glob.glob(os.path.join(XG_DIR, "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=";", encoding="utf-8")
        except Exception:
            df = pd.read_csv(f, sep=";", encoding="latin-1")

        df["Season"] = _parse_season_label(f)

        # Spalten umbenennen falls nötig
        df.columns = [c.strip() for c in df.columns]
        if "team" in df.columns:
            df = df.rename(columns={"team": "Team"})

        # Teamnamen normalisieren
        df["Team"] = df["Team"].apply(normalize_team)

        # xG per Spiel berechnen (aus Saisonsummen)
        if "matches" in df.columns:
            for col in ["xG", "xGA", "xPTS"]:
                if col in df.columns:
                    df[f"{col}_per_game"] = pd.to_numeric(df[col], errors="coerce") / pd.to_numeric(df["matches"], errors="coerce")
        dfs.append(df)

    xg_data = pd.concat(dfs, ignore_index=True)
    return xg_data


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING WINDOW FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_features(df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    """
    Fügt Rolling-Window-Features (EWA) für Heim- und Auswärtsteam hinzu.
    Berücksichtigt Overall-Form sowie spezifische Home- und Away-Form.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Bilde ein "Team-Spiel"-Format, bei dem jedes Spiel 2 Zeilen hat (1x pro Team)
    home_stats = df[["Season", "Date", "HomeTeam", "FTHG", "FTAG", "FTR", "HS", "HST"]].copy()
    home_stats.rename(columns={
        "HomeTeam": "Team", "FTHG": "GoalsScored", "FTAG": "GoalsConceded",
        "HS": "Shots", "HST": "ShotsOnTarget"
    }, inplace=True)
    home_stats["is_home"] = 1
    
    away_stats = df[["Season", "Date", "AwayTeam", "FTHG", "FTAG", "FTR", "AS", "AST"]].copy()
    away_stats.rename(columns={
        "AwayTeam": "Team", "FTAG": "GoalsScored", "FTHG": "GoalsConceded",
        "AS": "Shots", "AST": "ShotsOnTarget"
    }, inplace=True)
    away_stats["is_home"] = 0
    
    # Vereine und sortieren
    all_stats = pd.concat([home_stats, away_stats]).sort_values(["Date", "Team"]).reset_index(drop=True)
    
    # Punkte berechnen (aus Sicht des aktuellen Teams)
    def calc_points(row):
        if row["FTR"] == "D": return 1
        if row["is_home"] == 1 and row["FTR"] == "H": return 3
        if row["is_home"] == 0 and row["FTR"] == "A": return 3
        return 0
        
    all_stats["Points"] = all_stats.apply(calc_points, axis=1)
    
    def compute_ewa(data, span):
        grouped = data.groupby("Team")
        shifted = grouped.shift(1)  # Nimm die Werte der VERGANGENEN Spiele
        ewa = shifted[["GoalsScored", "GoalsConceded", "Points", "Shots", "ShotsOnTarget"]].groupby(data["Team"]).ewm(span=span, min_periods=1).mean()
        ewa = ewa.reset_index(level=0, drop=True)
        return ewa

    # --- 1. OVERALL FORM ---
    overall_ewa = compute_ewa(all_stats, span)
    overall_ewa.columns = [f"overall_ewa_{c}" for c in overall_ewa.columns]
    all_stats = pd.concat([all_stats, overall_ewa], axis=1)
    
    # --- 2. HOME FORM ---
    home_only = all_stats[all_stats["is_home"] == 1].copy()
    home_ewa = compute_ewa(home_only, span)
    home_ewa.columns = [f"home_form_ewa_{c}" for c in home_ewa.columns]
    home_only = pd.concat([home_only, home_ewa], axis=1)
    
    # --- 3. AWAY FORM ---
    away_only = all_stats[all_stats["is_home"] == 0].copy()
    away_ewa = compute_ewa(away_only, span)
    away_ewa.columns = [f"away_form_ewa_{c}" for c in away_ewa.columns]
    away_only = pd.concat([away_only, away_ewa], axis=1)
    
    # Zurück in das Haupt-Dataframe (df) mergen
    h_features = home_only[["Date", "Team"] + list(home_ewa.columns)]
    h_overall = all_stats[all_stats["is_home"] == 1][["Date", "Team"] + list(overall_ewa.columns)]
    h_all = h_features.merge(h_overall, on=["Date", "Team"], how="left")
    
    a_features = away_only[["Date", "Team"] + list(away_ewa.columns)]
    a_overall = all_stats[all_stats["is_home"] == 0][["Date", "Team"] + list(overall_ewa.columns)]
    a_all = a_features.merge(a_overall, on=["Date", "Team"], how="left")
    
    h_all.columns = [f"home_team_{c}" if c not in ["Date", "Team"] else c for c in h_all.columns]
    a_all.columns = [f"away_team_{c}" if c not in ["Date", "Team"] else c for c in a_all.columns]
    
    df = df.merge(h_all, left_on=["Date", "HomeTeam"], right_on=["Date", "Team"], how="left").drop(columns=["Team"])
    df = df.merge(a_all, left_on=["Date", "AwayTeam"], right_on=["Date", "Team"], how="left").drop(columns=["Team"])
    
    return df


def merge_xg_features(match_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt xG-Saisondaten (per Spiel normiert) als Features hinzu.
    Join über Team + Saison.

    [Update] Data-Leakage-Behebung:
    Für abgeschlossene Saisons: Vorjahres-xG als Prior (Lag 1).
    Für die maximale/aktuelle Saison: direkte Nutzung (kein Lag), da wöchentlich aktuell.
    """
    print("   📊 xG-Daten: Vorjahres-Lag für historische Saisons, direkte Nutzung für aktuelle Saison")
    xg_cols = ["Team", "Season", "xG_per_game", "xGA_per_game", "xPTS_per_game"]
    available = [c for c in xg_cols if c in xg_df.columns]
    xg_slim = xg_df[available].copy()

    all_seasons = sorted(xg_slim["Season"].unique())
    max_season = all_seasons[-1] if len(all_seasons) > 0 else None

    # Baue xg_mapped auf, indem wir pro xG-Saison definieren, für welche MatchSeason es gilt
    rows = []
    for i, s in enumerate(all_seasons):
        prior = all_seasons[i - 1] if i > 0 else s
        join_season = s if s == max_season else prior
        
        s_data = xg_slim[xg_slim["Season"] == join_season].copy()
        s_data["MatchSeason"] = s
        rows.append(s_data)
        
    xg_mapped = pd.concat(rows, ignore_index=True) if len(rows) > 0 else xg_slim.copy()

    # Home join (Match-Saison → ermittelte xG-Saison)
    home_xg = xg_mapped.drop(columns=["Season"]).rename(columns={
        "Team":          "HomeTeam",
        "MatchSeason":   "Season",
        "xG_per_game":   "home_xG_per_game",
        "xGA_per_game":  "home_xGA_per_game",
        "xPTS_per_game": "home_xPTS_per_game",
    })
    # Nur relevante Spalten behalten
    home_xg = home_xg[["HomeTeam", "Season", "home_xG_per_game", "home_xGA_per_game", "home_xPTS_per_game"]]
    result = match_df.merge(home_xg, on=["HomeTeam", "Season"], how="left")

    # Away join
    away_xg = xg_mapped.drop(columns=["Season"]).rename(columns={
        "Team":          "AwayTeam",
        "MatchSeason":   "Season",
        "xG_per_game":   "away_xG_per_game",
        "xGA_per_game":  "away_xGA_per_game",
        "xPTS_per_game": "away_xPTS_per_game",
    })
    away_xg = away_xg[["AwayTeam", "Season", "away_xG_per_game", "away_xGA_per_game", "away_xPTS_per_game"]]
    result = result.merge(away_xg, on=["AwayTeam", "Season"], how="left")

    return result



# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-LISTE
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "home_team_overall_ewa_GoalsScored", "home_team_overall_ewa_GoalsConceded",
    "home_team_overall_ewa_Points", "home_team_overall_ewa_Shots", "home_team_overall_ewa_ShotsOnTarget",
    "home_team_home_form_ewa_GoalsScored", "home_team_home_form_ewa_GoalsConceded",
    "home_team_home_form_ewa_Points", "home_team_home_form_ewa_Shots", "home_team_home_form_ewa_ShotsOnTarget",

    "away_team_overall_ewa_GoalsScored", "away_team_overall_ewa_GoalsConceded",
    "away_team_overall_ewa_Points", "away_team_overall_ewa_Shots", "away_team_overall_ewa_ShotsOnTarget",
    "away_team_away_form_ewa_GoalsScored", "away_team_away_form_ewa_GoalsConceded",
    "away_team_away_form_ewa_Points", "away_team_away_form_ewa_Shots", "away_team_away_form_ewa_ShotsOnTarget",

    "goal_diff_avg",       # home - away Durchschnitts-Tordifferenz
    "points_diff_avg",     # home - away Durchschnittspunkte

    # [NEU] Unentschieden-Features
    "abs_goal_diff",           # Absolutwert der Tordifferenz (je kleiner, desto ausgeglichener)
    "abs_points_diff",         # Absolutwert der Punktedifferenz
    "combined_draw_tendency",  # Kombinierter Indikator: niedrig = Gleichgewicht = Unentschieden wahrscheinlicher
]

XG_FEATURES = BASE_FEATURES + [
    "home_xG_per_game", "home_xGA_per_game", "home_xPTS_per_game",
    "away_xG_per_game", "away_xGA_per_game", "away_xPTS_per_game",
    "xG_diff", "xGA_diff",
    # [NEU] Unentschieden-Features (xG-spezifisch)
    "abs_xG_diff",             # Absolutwert der xG-Differenz
]


def add_derived_features(df: pd.DataFrame, include_xg: bool = False) -> pd.DataFrame:
    """Fügt abgeleitete Differenz-Features hinzu."""
    df = df.copy()
    df["goal_diff_avg"]   = df["home_team_overall_ewa_GoalsScored"] - df["away_team_overall_ewa_GoalsScored"]
    df["points_diff_avg"] = df["home_team_overall_ewa_Points"]       - df["away_team_overall_ewa_Points"]

    # [NEU] Unentschieden-Features
    # Absolutwerte der Differenzen (je kleiner, desto ausgeglichener das Spiel)
    df["abs_goal_diff"]   = df["goal_diff_avg"].abs()
    df["abs_points_diff"] = df["points_diff_avg"].abs()

    # Formstabilität: je niedriger beide Absolutdifferenzen, desto wahrscheinlicher Unentschieden
    df["combined_draw_tendency"] = (
        1.0 / (1.0 + df["abs_goal_diff"]) *
        1.0 / (1.0 + df["abs_points_diff"])
    )

    if include_xg:
        df["xG_diff"]  = df["home_xG_per_game"]  - df["away_xG_per_game"]
        df["xGA_diff"] = df["home_xGA_per_game"]  - df["away_xGA_per_game"]
        # [NEU] Unentschieden-Features (xG-spezifisch)
        df["abs_xG_diff"] = df["xG_diff"].abs()

    return df


def prepare_dataset(include_xg: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Hauptfunktion: Lädt alle Daten, berechnet Features und gibt
    den fertigen DataFrame + Feature-Liste zurück.
    """
    print("📂 Lade Bundesliga-Spieldaten ...")
    matches = load_bundesliga_data()
    print(f"   → {len(matches)} Spiele geladen ({matches['Season'].nunique()} Saisons)")

    print("🔄 Berechne Rolling-Window-Features (EWA) ...")
    matches = compute_rolling_features(matches)

    if include_xg:
        print("📊 Lade xG-Daten und füge sie hinzu ...")
        xg = load_xg_data()
        print(f"   → {len(xg)} Team-Saison-Einträge geladen")
        matches = merge_xg_features(matches, xg)
        matches = add_derived_features(matches, include_xg=True)
        feature_list = XG_FEATURES
    else:
        matches = add_derived_features(matches, include_xg=False)
        feature_list = BASE_FEATURES

    # Nur Zeilen mit vollständigen Features
    available_features = [f for f in feature_list if f in matches.columns]
    df_clean = matches.dropna(subset=available_features + ["FTR"])

    print(f"✅ Dataset bereit: {len(df_clean)} Spiele mit vollständigen Features")
    print(f"   Features ({len(available_features)}): {available_features}")

    return df_clean, available_features
