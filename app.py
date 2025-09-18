# -*- coding: utf-8 -*-
"""Predicting Football Matches
"""

#imports - pandas/numpy for data, requests for api, xgboost + sklearn for model
import pandas as pd, numpy as np, requests, time, json, math, datetime as dt
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import streamlit as st

#Caching helpers
@st.cache_data(show_spinner=True)
def fetch_bootstrap_and_fixtures():
    boot = get_json(f"{BASE}/bootstrap-static/")
    players_meta = pd.DataFrame(boot["elements"])
    teams_meta   = pd.DataFrame(boot["teams"])
    fixtures     = pd.DataFrame(get_json(f"{BASE}/fixtures/"))
    return players_meta, teams_meta, fixtures

@st.cache_data(show_spinner=True)
def tidy_meta_cached(players_meta, teams_meta):
    teams = teams_meta.rename(columns={"id":"team_id"})[
        ["team_id","name","short_name","strength",
         "strength_attack_home","strength_attack_away",
         "strength_defence_home","strength_defence_away"]
    ]
    players = players_meta.rename(columns={"id":"player_id","team":"team_id"})[
        ["player_id","first_name","second_name","web_name","team_id","element_type"]
    ].merge(teams, on="team_id", how="left")
    return players, teams

@st.cache_data(show_spinner=True)
def fetch_all_histories_limited(player_ids, limit=None):
    ids = player_ids if limit is None else player_ids[:limit]
    out = []
    for pid in ids:
        try:
            j = get_json(f"{BASE}/element-summary/{pid}/")
            df = pd.DataFrame(j.get("history", []))
            if df.empty: 
                continue
            needed = ['element','opponent_team','round','minutes','total_points','goals_scored','assists',
                      'ict_index','creativity','influence','threat',
                      'expected_goals','expected_assists','expected_goal_involvements',
                      'expected_goals_conceded','was_home','kickoff_time']
            for c in needed:
                if c not in df.columns: df[c] = np.nan
            df["player_id"] = pid
            out.append(df)
        except:
            pass
    hist = pd.concat(out, ignore_index=True)
    hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"], errors="coerce")
    hist["round"] = pd.to_numeric(hist["round"], errors="coerce")
    hist["was_home"] = hist["was_home"].astype("Int64")
    hist = hist[hist["kickoff_time"].notna()].sort_values(["player_id","kickoff_time"]).reset_index(drop=True)
    return hist

@st.cache_resource(show_spinner=True)
def train_model_cached(X, y):
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    model.fit(X, y)
    return model

# Streamlit setup
st.set_page_config(page_title="EPL Player Predictor", page_icon="⚽", layout="wide")
st.title("⚽ EPL Player Predictor")

pd.set_option("display.max_columns", 200)

# function to grab json data from the FPL api with a retry
BASE = "https://fantasy.premierleague.com/api"

def get_json(url, retries=5, sleep=0.5):
    for i in range(retries):
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(sleep*(i+1))
    r.raise_for_status()

# controls to avoid heavy work at import
st.sidebar.header("Run options")
limit_partial = st.sidebar.checkbox("Use partial dataset (faster first run)", value=True)
max_players   = st.sidebar.slider("Max players when partial", 50, 600, 200, 50)
build = st.button("Build / Refresh predictions")

if build:
    try:
        # 1) Grab static data
        bootstrap = get_json(f"{BASE}/bootstrap-static/")
        players_meta = pd.DataFrame(bootstrap['elements'])
        teams_meta   = pd.DataFrame(bootstrap['teams'])
        fixtures     = pd.DataFrame(get_json(f"{BASE}/fixtures/"))

        # 2) Compact teams & players
        teams = teams_meta[['id','name','short_name','strength',
                            'strength_attack_home','strength_attack_away',
                            'strength_defence_home','strength_defence_away']].rename(columns={'id':'team_id'})

        players = players_meta[['id','first_name','second_name','web_name','team','element_type']] \
                    .rename(columns={'id':'player_id','team':'team_id'}) \
                    .merge(teams, on='team_id', how='left')

        st.subheader("Players (sample)")
        st.dataframe(players.head(10), use_container_width=True)

        # 3) Match histories (respect partial setting for faster runs)
        all_hist = []
        ids = players['player_id'].tolist()
        if limit_partial:
            ids = ids[:max_players]

        for pid in tqdm(ids, desc="fetching players"):
            try:
                h = fetch_player_history(pid)
                if not h.empty:
                    all_hist.append(h)
            except:
                pass  # skip failures

        if not all_hist:
            st.error("No player histories were fetched. Try unchecking partial dataset or increase max players.")
            st.stop()

        hist = pd.concat(all_hist, ignore_index=True)
        hist['kickoff_time'] = pd.to_datetime(hist['kickoff_time'], errors='coerce')
        hist['round'] = pd.to_numeric(hist['round'], errors='coerce')
        hist['was_home'] = hist['was_home'].astype('Int64')
        hist = hist[hist['kickoff_time'].notna()] \
                   .sort_values(['player_id','kickoff_time']) \
                   .reset_index(drop=True)

        st.subheader("History (sample)")
        st.dataframe(hist.head(10), use_container_width=True)

        # 4) Opponent context
        opp = teams.rename(columns={
            'team_id':'opp_team_id','name':'opp_name','short_name':'opp_short_name',
            'strength':'opp_strength',
            'strength_defence_home':'opp_strength_defence_home',
            'strength_defence_away':'opp_strength_defence_away'
        })

        hist = hist.merge(
            players[['player_id','team_id','web_name','element_type',
                     'strength','strength_attack_home','strength_attack_away',
                     'strength_defence_home','strength_defence_away']],
            on='player_id', how='left'
        ).merge(
            opp[['opp_team_id','opp_strength','opp_strength_defence_home','opp_strength_defence_away']],
            left_on='opponent_team', right_on='opp_team_id', how='left'
        )

        hist['team_strength_diff'] = hist['strength'] - hist['opp_strength']

        # 5) Feature engineering
        def add_player_features(df, lags=(1,2,3), windows=(3,5,8)):
            df = df.copy()
            grp = df.groupby('player_id', group_keys=False)
            base_cols = ['total_points','minutes','goals_scored','assists',
                         'ict_index','creativity','influence','threat',
                         'expected_goals','expected_assists','expected_goal_involvements']

            for col in base_cols:
                for L in lags:
                    df[f'{col}_lag{L}'] = grp[col].shift(L)

            for W in windows:
                for col in base_cols:
                    df[f'{col}_roll{W}_mean'] = grp[col].shift(1).rolling(W).mean()
                    df[f'{col}_roll{W}_sum']  = grp[col].shift(1).rolling(W).sum()

            df['played_last_match'] = grp['minutes'].shift(1).fillna(0).gt(0).astype(int)
            df['played_last3_pct']  = grp['minutes'].shift(1).rolling(3).apply(lambda x: np.mean(x>0), raw=True)

            df['attack_v_def_diff'] = np.where(
                df['was_home']==1,
                df['strength_attack_home'] - df['opp_strength_defence_away'],
                df['strength_attack_away'] - df['opp_strength_defence_home']
            )

            df['month'] = df['kickoff_time'].dt.month
            df['dow'] = df['kickoff_time'].dt.dayofweek
            return df

        fe = add_player_features(hist)
        fe['y_next_points'] = fe.groupby('player_id')['total_points'].shift(-1)

        model_df = fe.dropna(subset=['y_next_points','total_points_lag1','minutes_lag1']).copy()

        exclude = {'y_next_points','total_points','kickoff_time','web_name','opp_name','opp_short_name',
                   'opp_team_id','team_id','opponent_team','name','short_name'}
        feature_cols = [c for c in model_df.columns
                        if c not in exclude and c != 'was_home' and pd.api.types.is_numeric_dtype(model_df[c])]

        X = model_df[feature_cols].fillna(0)
        y = model_df['y_next_points'].astype(float)

        # 6) Train final model (simple fit; you already validated offline)
        final_model = XGBRegressor(
            n_estimators=800, learning_rate=0.04, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1, tree_method="hist"
        )
        final_model.fit(X, y, verbose=False)

        # 7) Build next-GW frame & predict
        upcoming = fixtures.copy()
        next_gw = upcoming.loc[(~upcoming['finished']) & (upcoming['event'].notna()), 'event'].min()
        if pd.isna(next_gw):
            st.warning("No upcoming gameweek found yet — try again later.")
            st.stop()

        upcoming_next = upcoming[(upcoming['event']==next_gw) & (~upcoming['finished'])].copy()

        home = upcoming_next[['team_h','team_a']].rename(columns={'team_h':'team_id','team_a':'opp_team_id'}).assign(was_home=1)
        away = upcoming_next[['team_a','team_h']].rename(columns={'team_a':'team_id','team_h':'opp_team_id'}).assign(was_home=0)
        team_next = pd.concat([home, away], ignore_index=True)

        latest = fe.sort_values(['player_id','kickoff_time']).groupby('player_id').tail(1).copy()
        latest = latest.merge(players[['player_id','team_id','web_name','element_type',
                                       'strength_attack_home','strength_attack_away',
                                       'strength_defence_home','strength_defence_away']],
                              on='player_id', how='left')
        latest = latest.merge(team_next, on='team_id', how='left')

        opp2 = teams.rename(columns={'team_id':'opp_team_id','strength':'opp_strength',
                                     'strength_defence_home':'opp_strength_defence_home',
                                     'strength_defence_away':'opp_strength_defence_away'})
        latest = latest.merge(opp2[['opp_team_id','opp_strength','opp_strength_defence_home','opp_strength_defence_away']],
                              on='opp_team_id', how='left')

        latest['attack_v_def_diff'] = np.where(
            latest['was_home'].eq(1),
            latest['strength_attack_home'] - latest['opp_strength_defence_away'],
            latest['strength_attack_away'] - latest['opp_strength_defence_home']
        )

        X_pred = latest.reindex(columns=feature_cols).fillna(0)
        latest['pred_next_points'] = final_model.predict(X_pred)

        TEAM_MAP = teams.set_index("team_id")["short_name"].to_dict()
        POS_MAP  = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}
        latest['team'] = latest['team_id'].map(TEAM_MAP)
        latest['opp']  = latest['opp_team_id'].map(TEAM_MAP)
        latest['position'] = latest['element_type'].map(POS_MAP)
        latest['pred_next_points'] = latest['pred_next_points'].round(2)

        # 8) Filters + display
        st.sidebar.header("Filters")
        pos_choice  = st.sidebar.selectbox("Position", ["All","GK","DEF","MID","FWD"])
        team_choice = st.sidebar.selectbox("Team", ["All"] + sorted(set(TEAM_MAP.values())))
        min_points  = st.sidebar.slider("Minimum predicted points", 0.0, 15.0, 4.0, 0.5)
        top_n       = st.sidebar.slider("How many players to display", 10, 300, 50, 10)
        search      = st.sidebar.text_input("Search player name")

        tbl = latest[['web_name','position','team','opp','was_home','pred_next_points']].copy()
        if pos_choice != "All":
            tbl = tbl[tbl['position'] == pos_choice]
        if team_choice != "All":
            tbl = tbl[tbl['team'] == team_choice]
        tbl = tbl[tbl['pred_next_points'] >= min_points]
        if search.strip():
            s = search.strip().lower()
            tbl = tbl[tbl['web_name'].str.lower().str.contains(s)]

        tbl = tbl.sort_values('pred_next_points', ascending=False).reset_index(drop=True)

        st.subheader(f"Predictions — Gameweek {int(next_gw)}")
        st.dataframe(tbl.head(top_n), use_container_width=True)

        st.download_button(
            label="Download predictions as CSV",
            data=tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"epl_predictions_gw{int(next_gw)}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Something went wrong while building predictions.")
        st.exception(e)

else:
    st.info("Click **Build / Refresh predictions** to start.")


