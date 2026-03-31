import os
import numpy as np
import pandas as pd


def load_and_prepare_data():
    old_file = "data/matches.csv"
    new_file = "data/epl_matches.csv"
    dataframes = []

    rename_map = {
        "FTH Goals": "FTHG",
        "FTA Goals": "FTAG",
        "FT Result": "FTR",
        "H Shots": "HS",
        "A Shots": "AS",
        "H SOT": "HST",
        "A SOT": "AST",
    }

    if os.path.exists(old_file):
        df_old = pd.read_csv(old_file)
        df_old.columns = df_old.columns.str.strip()
        df_old = df_old.rename(columns=rename_map)
        dataframes.append(df_old)

    if os.path.exists(new_file):
        df_new = pd.read_csv(new_file)
        df_new.columns = df_new.columns.str.strip()
        df_new = df_new.rename(columns=rename_map)
        if "Season" not in df_new.columns:
            df_new["Season"] = "2025/26"
        dataframes.append(df_new)

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.dropna(subset=["Season", "HomeTeam", "AwayTeam"])
    combined_df["Season"] = combined_df["Season"].astype(str)
    combined_df["Date"] = pd.to_datetime(combined_df["Date"], dayfirst=True, errors="coerce")
    combined_df = combined_df.dropna(subset=["Date"])

    numeric_cols = [col for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST"] if col in combined_df.columns]
    for col in numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    return combined_df.sort_values("Date").reset_index(drop=True)


def create_team_features(df, season_filter=None):
    """Legacy static team features kept for app.py compatibility."""
    if season_filter:
        df = df[df["Season"] == season_filter].copy()

    if df.empty:
        return pd.DataFrame()

    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    features = []

    for team in teams:
        team_df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")

        scored = team_df.apply(lambda x: x["FTHG"] if x["HomeTeam"] == team else x["FTAG"], axis=1)
        conceded = team_df.apply(lambda x: x["FTAG"] if x["HomeTeam"] == team else x["FTHG"], axis=1)

        recent_scored = scored.iloc[-5:].mean() if len(scored) >= 5 else scored.mean()
        recent_conceded = conceded.iloc[-5:].mean() if len(conceded) >= 5 else conceded.mean()

        home_games = team_df[team_df["HomeTeam"] == team]
        away_games = team_df[team_df["AwayTeam"] == team]

        features.append(
            {
                "Team": team,
                "AvgGoalsScored": recent_scored,
                "AvgGoalsConceded": recent_conceded,
                "HomeAttack": home_games["FTHG"].mean() if not home_games.empty else recent_scored,
                "AwayAttack": away_games["FTAG"].mean() if not away_games.empty else recent_scored,
                "HomeDefense": home_games["FTAG"].mean() if not home_games.empty else recent_conceded,
                "AwayDefense": away_games["FTHG"].mean() if not away_games.empty else recent_conceded,
            }
        )

    return pd.DataFrame(features)


def create_match_level_features(df, window=5, min_history_matches=3):
    """
    Build leakage-safe, per-match features.
    Every feature for a match uses only each team's previous matches.
    """
    required = {"Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if required - set(df.columns):
        return pd.DataFrame()

    match_df = df.copy()
    match_df = match_df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    match_df = match_df[match_df["FTR"].isin(["H", "D", "A"])].copy()
    if match_df.empty:
        return pd.DataFrame()

    match_df = match_df.sort_values("Date").reset_index(drop=True)
    match_df["MatchId"] = match_df.index

    hs = match_df["HS"] if "HS" in match_df.columns else np.nan
    a_s = match_df["AS"] if "AS" in match_df.columns else np.nan
    hst = match_df["HST"] if "HST" in match_df.columns else np.nan
    ast = match_df["AST"] if "AST" in match_df.columns else np.nan

    home_points = np.where(match_df["FTR"] == "H", 3, np.where(match_df["FTR"] == "D", 1, 0))
    away_points = np.where(match_df["FTR"] == "A", 3, np.where(match_df["FTR"] == "D", 1, 0))

    home_rows = pd.DataFrame(
        {
            "MatchId": match_df["MatchId"],
            "Date": match_df["Date"],
            "Team": match_df["HomeTeam"],
            "IsHome": 1,
            "GoalsFor": match_df["FTHG"],
            "GoalsAgainst": match_df["FTAG"],
            "ShotsFor": hs,
            "ShotsAgainst": a_s,
            "SOTFor": hst,
            "SOTAgainst": ast,
            "Points": home_points,
        }
    )

    away_rows = pd.DataFrame(
        {
            "MatchId": match_df["MatchId"],
            "Date": match_df["Date"],
            "Team": match_df["AwayTeam"],
            "IsHome": 0,
            "GoalsFor": match_df["FTAG"],
            "GoalsAgainst": match_df["FTHG"],
            "ShotsFor": a_s,
            "ShotsAgainst": hs,
            "SOTFor": ast,
            "SOTAgainst": hst,
            "Points": away_points,
        }
    )

    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_rows = team_rows.sort_values(["Team", "Date", "MatchId"]).reset_index(drop=True)

    base_metrics = ["GoalsFor", "GoalsAgainst", "Points", "ShotsFor", "ShotsAgainst", "SOTFor", "SOTAgainst"]
    for metric in base_metrics:
        team_rows[f"Form_{metric}"] = (
            team_rows.groupby("Team")[metric]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        )

    team_rows["Form_GoalDiff"] = team_rows["Form_GoalsFor"] - team_rows["Form_GoalsAgainst"]
    team_rows["MatchesPlayedBefore"] = team_rows.groupby("Team").cumcount()

    feature_cols = [
        "Form_GoalsFor",
        "Form_GoalsAgainst",
        "Form_GoalDiff",
        "Form_Points",
        "Form_ShotsFor",
        "Form_ShotsAgainst",
        "Form_SOTFor",
        "Form_SOTAgainst",
        "MatchesPlayedBefore",
    ]

    home_features = team_rows[team_rows["IsHome"] == 1][["MatchId"] + feature_cols].rename(
        columns={col: f"H_{col}" for col in feature_cols}
    )
    away_features = team_rows[team_rows["IsHome"] == 0][["MatchId"] + feature_cols].rename(
        columns={col: f"A_{col}" for col in feature_cols}
    )

    model_df = match_df.merge(home_features, on="MatchId", how="left")
    model_df = model_df.merge(away_features, on="MatchId", how="left")

    if min_history_matches > 0:
        model_df = model_df[
            (model_df["H_MatchesPlayedBefore"] >= min_history_matches)
            & (model_df["A_MatchesPlayedBefore"] >= min_history_matches)
        ].copy()

    model_df["Result_Encoded"] = model_df["FTR"].map({"H": 0, "D": 1, "A": 2})

    model_df["Diff_Form_GoalsFor"] = model_df["H_Form_GoalsFor"] - model_df["A_Form_GoalsFor"]
    model_df["Diff_Form_GoalsAgainst"] = model_df["H_Form_GoalsAgainst"] - model_df["A_Form_GoalsAgainst"]
    model_df["Diff_Form_GoalDiff"] = model_df["H_Form_GoalDiff"] - model_df["A_Form_GoalDiff"]
    model_df["Diff_Form_Points"] = model_df["H_Form_Points"] - model_df["A_Form_Points"]
    model_df["Diff_Form_ShotsFor"] = model_df["H_Form_ShotsFor"] - model_df["A_Form_ShotsFor"]
    model_df["Diff_Form_SOTFor"] = model_df["H_Form_SOTFor"] - model_df["A_Form_SOTFor"]
    # Smaller absolute gaps often correlate with draw-like matches.
    model_df["AbsDiff_Form_GoalsFor"] = model_df["Diff_Form_GoalsFor"].abs()
    model_df["AbsDiff_Form_GoalsAgainst"] = model_df["Diff_Form_GoalsAgainst"].abs()
    model_df["AbsDiff_Form_GoalDiff"] = model_df["Diff_Form_GoalDiff"].abs()
    model_df["AbsDiff_Form_Points"] = model_df["Diff_Form_Points"].abs()
    model_df["AbsDiff_Form_ShotsFor"] = model_df["Diff_Form_ShotsFor"].abs()
    model_df["AbsDiff_Form_SOTFor"] = model_df["Diff_Form_SOTFor"].abs()

    return model_df.sort_values("Date").reset_index(drop=True)
