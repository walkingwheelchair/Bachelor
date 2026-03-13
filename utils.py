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

ROLLING_N = 20  # letzte N Spiele


def _get_team_match_history(df: pd.DataFrame) -> dict:
    """
    Erstellt für jedes Team eine chronologisch sortierte Liste ihrer Spiele
    mit relevanten Statistiken (aus Heim- UND Auswärtsspielen).
    Gibt dict: team -> list of dicts
    """
    history = {}  # team -> [ {date, goals_scored, goals_conceded, points}, ... ]

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        ftr = row["FTR"]
        date = row.get("Date", pd.NaT)

        fthg = row.get("FTHG", np.nan)
        ftag = row.get("FTAG", np.nan)

        # Punkte
        if ftr == "H":
            home_pts, away_pts = 3, 0
        elif ftr == "A":
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1

        for team, scored, conceded, pts, is_home in [
            (home, fthg, ftag, home_pts, 1),
            (away, ftag, fthg, away_pts, 0),
        ]:
            if team not in history:
                history[team] = []
            history[team].append({
                "date": date,
                "goals_scored": scored,
                "goals_conceded": conceded,
                "points": pts,
                "is_home": is_home,
                "shots": row.get("HS" if is_home else "AS", np.nan),
                "shots_on_target": row.get("HST" if is_home else "AST", np.nan),
            })

    return history


def compute_rolling_features(df: pd.DataFrame, n: int = ROLLING_N) -> pd.DataFrame:
    """
    Fügt Rolling-Window-Features (letzte N Spiele) für Heim- und Auswärtsteam hinzu.
    Features werden aus den Spielen VOR dem aktuellen Spiel berechnet (kein Data-Leakage).
    """
    # Sortiere nach Datum
    df = df.sort_values("Date").reset_index(drop=True)

    # Initialisiere Spalten
    feature_cols = [
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "home_avg_points", "home_win_rate", "home_draw_rate", "home_loss_rate",
        "home_avg_shots", "home_avg_shots_on_target",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "away_avg_points", "away_win_rate", "away_draw_rate", "away_loss_rate",
        "away_avg_shots", "away_avg_shots_on_target",
    ]
    for col in feature_cols:
        df[col] = np.nan

    # Team-History aufbauen (wird iterativ befüllt)
    team_history: dict = {}

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Features aus bisheriger History berechnen
        for team, prefix in [(home, "home"), (away, "away")]:
            hist = team_history.get(team, [])
            last_n = hist[-n:] if len(hist) >= 1 else []

            if len(last_n) == 0:
                continue  # keine History → NaN bleibt

            goals_scored    = [g["goals_scored"] for g in last_n if not np.isnan(g.get("goals_scored", np.nan))]
            goals_conceded  = [g["goals_conceded"] for g in last_n if not np.isnan(g.get("goals_conceded", np.nan))]
            points          = [g["points"] for g in last_n]
            shots           = [g["shots"] for g in last_n if not np.isnan(g.get("shots", np.nan))]
            shots_on_target = [g["shots_on_target"] for g in last_n if not np.isnan(g.get("shots_on_target", np.nan))]

            if goals_scored:
                df.at[idx, f"{prefix}_avg_goals_scored"]    = np.mean(goals_scored)
            if goals_conceded:
                df.at[idx, f"{prefix}_avg_goals_conceded"]  = np.mean(goals_conceded)
            if points:
                df.at[idx, f"{prefix}_avg_points"]          = np.mean(points)
                df.at[idx, f"{prefix}_win_rate"]             = sum(p == 3 for p in points) / len(points)
                df.at[idx, f"{prefix}_draw_rate"]            = sum(p == 1 for p in points) / len(points)
                df.at[idx, f"{prefix}_loss_rate"]            = sum(p == 0 for p in points) / len(points)
            if shots:
                df.at[idx, f"{prefix}_avg_shots"]           = np.mean(shots)
            if shots_on_target:
                df.at[idx, f"{prefix}_avg_shots_on_target"] = np.mean(shots_on_target)

        # Jetzt das aktuelle Spiel zur History hinzufügen
        ftr   = row["FTR"]
        fthg  = row.get("FTHG", np.nan)
        ftag  = row.get("FTAG", np.nan)
        h_pts = 3 if ftr == "H" else (1 if ftr == "D" else 0)
        a_pts = 3 if ftr == "A" else (1 if ftr == "D" else 0)

        for team, scored, conceded, pts, is_home in [
            (home, fthg, ftag, h_pts, 1),
            (away, ftag, fthg, a_pts, 0),
        ]:
            if team not in team_history:
                team_history[team] = []
            team_history[team].append({
                "goals_scored":    scored,
                "goals_conceded":  conceded,
                "points":          pts,
                "is_home":         is_home,
                "shots":           row.get("HS" if is_home else "AS", np.nan),
                "shots_on_target": row.get("HST" if is_home else "AST", np.nan),
            })

    return df


def merge_xg_features(match_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt xG-Saisondaten (per Spiel normiert) als Features hinzu.
    Join über Team + Saison.
    """
    xg_cols = ["Team", "Season", "xG_per_game", "xGA_per_game", "xPTS_per_game"]
    available = [c for c in xg_cols if c in xg_df.columns]
    xg_slim = xg_df[available].copy()

    # Home join
    home_xg = xg_slim.rename(columns={
        "Team": "HomeTeam",
        "xG_per_game":   "home_xG_per_game",
        "xGA_per_game":  "home_xGA_per_game",
        "xPTS_per_game": "home_xPTS_per_game",
    })
    result = match_df.merge(home_xg, on=["HomeTeam", "Season"], how="left")

    # Away join
    away_xg = xg_slim.rename(columns={
        "Team": "AwayTeam",
        "xG_per_game":   "away_xG_per_game",
        "xGA_per_game":  "away_xGA_per_game",
        "xPTS_per_game": "away_xPTS_per_game",
    })
    result = result.merge(away_xg, on=["AwayTeam", "Season"], how="left")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-LISTE
# ─────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_avg_points", "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_avg_shots", "home_avg_shots_on_target",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_avg_points", "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_avg_shots", "away_avg_shots_on_target",
    "goal_diff_avg",       # home - away Durchschnitts-Tordifferenz
    "points_diff_avg",     # home - away Durchschnittspunkte
]

XG_FEATURES = BASE_FEATURES + [
    "home_xG_per_game", "home_xGA_per_game", "home_xPTS_per_game",
    "away_xG_per_game", "away_xGA_per_game", "away_xPTS_per_game",
    "xG_diff", "xGA_diff",
]


def add_derived_features(df: pd.DataFrame, include_xg: bool = False) -> pd.DataFrame:
    """Fügt abgeleitete Differenz-Features hinzu."""
    df = df.copy()
    df["goal_diff_avg"]   = df["home_avg_goals_scored"] - df["away_avg_goals_scored"]
    df["points_diff_avg"] = df["home_avg_points"]       - df["away_avg_points"]

    if include_xg:
        df["xG_diff"]  = df["home_xG_per_game"]  - df["away_xG_per_game"]
        df["xGA_diff"] = df["home_xGA_per_game"]  - df["away_xGA_per_game"]

    return df


def prepare_dataset(include_xg: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """
    Hauptfunktion: Lädt alle Daten, berechnet Features und gibt
    den fertigen DataFrame + Feature-Liste zurück.
    """
    print("📂 Lade Bundesliga-Spieldaten ...")
    matches = load_bundesliga_data()
    print(f"   → {len(matches)} Spiele geladen ({matches['Season'].nunique()} Saisons)")

    print("🔄 Berechne Rolling-Window-Features (letzte 20 Spiele) ...")
    matches = compute_rolling_features(matches, n=ROLLING_N)

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
