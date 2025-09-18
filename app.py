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

if build:
# grab static data: players, teams, fixtures
bootstrap = get_json(f"{BASE}/bootstrap-static/")
players_meta = pd.DataFrame(bootstrap['elements'])
teams_meta   = pd.DataFrame(bootstrap['teams'])
fixtures     = pd.DataFrame(get_json(f"{BASE}/fixtures/"))

# keep only useful team cols
teams = teams_meta[['id','name','short_name','strength',
                    'strength_attack_home','strength_attack_away',
                    'strength_defence_home','strength_defence_away']].rename(columns={'id':'team_id'})

# minimal player info
players = players_meta[['id','first_name','second_name','web_name','team','element_type']] \
            .rename(columns={'id':'player_id','team':'team_id'}) \
            .merge(teams, on='team_id', how='left')

players.head()

# --- controls to avoid heavy work at import ---
st.sidebar.header("Run options")
limit_partial = st.sidebar.checkbox("Use partial dataset (faster first run)", value=True)
max_players   = st.sidebar.slider("Max players when partial", 50, 600, 200, 50)

build = st.button("Build / Refresh predictions")

# function to get match-by-match history for a player
def fetch_player_history(pid):
    j = get_json(f"{BASE}/element-summary/{pid}/")
    df = pd.DataFrame(j.get('history', []))
    if df.empty:
        return df
    needed = ['element','opponent_team','round','minutes','total_points','goals_scored','assists',
              'ict_index','creativity','influence','threat',
              'expected_goals','expected_assists','expected_goal_involvements',
              'expected_goals_conceded','was_home','kickoff_time']
    for c in needed:
        if c not in df.columns: df[c] = np.nan
    df['player_id'] = pid
    return df

# loop over all players and get their match history
all_hist = []
for pid in tqdm(players['player_id'], desc="fetching players"):
    try:
        h = fetch_player_history(pid)
        if not h.empty: all_hist.append(h)
    except:
        pass  # if one player fails, skip

hist = pd.concat(all_hist, ignore_index=True)
hist['kickoff_time'] = pd.to_datetime(hist['kickoff_time'], errors='coerce')
hist['round'] = pd.to_numeric(hist['round'], errors='coerce')
hist['was_home'] = hist['was_home'].astype('Int64')
hist = hist[hist['kickoff_time'].notna()].sort_values(['player_id','kickoff_time']).reset_index(drop=True)

hist.head()

# add opponent info (strength etc.)
opp = teams.rename(columns={'team_id':'opp_team_id','name':'opp_name','short_name':'opp_short_name',
                            'strength':'opp_strength',
                            'strength_defence_home':'opp_strength_defence_home',
                            'strength_defence_away':'opp_strength_defence_away'})

hist = hist.merge(players[['player_id','team_id','web_name','element_type',
                           'strength','strength_attack_home','strength_attack_away',
                           'strength_defence_home','strength_defence_away']],
                  on='player_id', how='left')

hist = hist.merge(opp[['opp_team_id','opp_strength','opp_strength_defence_home','opp_strength_defence_away']],
                  left_on='opponent_team', right_on='opp_team_id', how='left')

hist['team_strength_diff'] = hist['strength'] - hist['opp_strength']
hist[['web_name','round','total_points','minutes','team_strength_diff']].head(10)

# add lag + rolling features so model can see "recent form"
def add_player_features(df, lags=(1,2,3), windows=(3,5,8)):
    df = df.copy()
    grp = df.groupby('player_id', group_keys=False)
    base_cols = ['total_points','minutes','goals_scored','assists',
                 'ict_index','creativity','influence','threat',
                 'expected_goals','expected_assists','expected_goal_involvements']

    # lag features
    for col in base_cols:
        for L in lags:
            df[f'{col}_lag{L}'] = grp[col].shift(L)

    # rolling means/sums
    for W in windows:
        for col in base_cols:
            df[f'{col}_roll{W}_mean'] = grp[col].shift(1).rolling(W).mean()
            df[f'{col}_roll{W}_sum']  = grp[col].shift(1).rolling(W).sum()

    # availability
    df['played_last_match'] = grp['minutes'].shift(1).fillna(0).gt(0).astype(int)
    df['played_last3_pct']  = grp['minutes'].shift(1).rolling(3).apply(lambda x: np.mean(x>0), raw=True)

    # attack vs defence diff
    df['attack_v_def_diff'] = np.where(
        df['was_home']==1,
        df['strength_attack_home'] - df['opp_strength_defence_away'],
        df['strength_attack_away'] - df['opp_strength_defence_home']
    )

    # time features
    df['month'] = df['kickoff_time'].dt.month
    df['dow'] = df['kickoff_time'].dt.dayofweek
    return df

fe = add_player_features(hist)

# we want to predict NEXT match total_points
fe['y_next_points'] = fe.groupby('player_id')['total_points'].shift(-1)

# drop rows that don’t have enough history/future
model_df = fe.dropna(subset=['y_next_points','total_points_lag1','minutes_lag1']).copy()
model_df.shape

exclude = {'y_next_points','total_points','kickoff_time','web_name','opp_name','opp_short_name',
           'opp_team_id','team_id','opponent_team','name','short_name'}
feature_cols = [c for c in model_df.columns if c not in exclude and c != 'was_home'
                and pd.api.types.is_numeric_dtype(model_df[c])]

X = model_df[feature_cols].fillna(0)
y = model_df['y_next_points'].astype(float)
groups = model_df['player_id']

# baseline = 3 game avg
baseline = model_df['total_points_roll3_mean'].fillna(model_df['total_points_lag1'])

# groupkfold so same player doesn't leak train/val
gkf = GroupKFold(n_splits=5)
oof_pred = np.zeros(len(model_df))

for tr, va in gkf.split(X, y, groups):
    model = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    oof_pred[va] = model.predict(X.iloc[va])

print("Model MAE:", mean_absolute_error(y, oof_pred))
print("Baseline MAE:", mean_absolute_error(y, baseline))

final_model = XGBRegressor(
    n_estimators=800, learning_rate=0.04, max_depth=6,
    subsample=0.9, colsample_bytree=0.9,
    random_state=42, n_jobs=-1, tree_method="hist"
)
final_model.fit(X, y, verbose=False)

# get latest row per player
latest = fe.sort_values(['player_id','kickoff_time']).groupby('player_id').tail(1).copy()

# Add print statement to inspect latest columns before merge with players
print("Columns of latest before merge with players:")
print(latest.columns)
print("\nHead of latest before merge with players:")
display(latest.head())


# Merge latest with players to get team_id and other player info
latest = latest.merge(players[['player_id','team_id','web_name','element_type',
                               'strength','strength_attack_home','strength_attack_away',
                               'strength_defence_home','strength_defence_away']],
                      on='player_id', how='left')

# Add print statement to inspect latest columns after merge with players
print("\nColumns of latest after merge with players:")
print(latest.columns)
print("\nHead of latest after merge with players:")
display(latest.head())


# next gameweek fixtures
upcoming = fixtures.copy()
next_gw = upcoming.loc[(~upcoming['finished']) & (upcoming['event'].notna()), 'event'].min()
upcoming_next = upcoming[(upcoming['event']==next_gw) & (~upcoming['finished'])].copy()

# team vs opponent
home = upcoming_next[['team_h','team_a']].rename(columns={'team_h':'team_id','team_a':'opp_team_id'}).assign(was_home=1)
away = upcoming_next[['team_a','team_h']].rename(columns={'team_a':'team_id','team_h':'opp_team_id'}).assign(was_home=0)
team_next = pd.concat([home,away])


# attach team_next to players' latest data based on the player's team_id
# Use left_on to specify the column from 'latest' and right_on for the column from 'team_next'
latest2 = latest.merge(team_next, left_on='team_id_x', right_on='team_id', how='left')

# Add print statement to inspect latest2 columns after merge with team_next
print("\nColumns of latest2 after merge with team_next:")
print(latest2.columns)
print("\nHead of latest2 after merge with team_next:")
display(latest2.head())


# recompute attack vs def for the new fixture
opp = teams.rename(columns={'team_id':'opp_team_id','strength':'opp_strength',
                            'strength_defence_home':'opp_strength_defence_home',
                            'strength_defence_away':'opp_strength_defence_away'})
# Use the correct column name 'opp_team_id_y' for the merge
latest2 = latest2.merge(opp[['opp_team_id','opp_strength','opp_strength_defence_home','opp_strength_defence_away']],
                        left_on='opp_team_id_y', right_on='opp_team_id', how='left')

latest2['attack_v_def_diff'] = np.where(
    latest2['was_home_y']==1, # Use was_home_y from the first merge
    latest2['strength_attack_home_x'] - latest2['opp_strength_defence_away_y'], # Use suffixed columns
    latest2['strength_attack_away_x'] - latest2['opp_strength_defence_home_y'] # Use suffixed columns
)

# predict
# Ensure X_pred has the same columns as X used for training
X_pred = latest2.reindex(columns=feature_cols).fillna(0)
latest2['pred_next_points'] = final_model.predict(X_pred)

# show top 20
# Use the correct column names for the final display
latest2[['web_name_x','team_id_x','opp_team_id_y','was_home_y','pred_next_points']].sort_values('pred_next_points', ascending=False).head(20).rename(columns={'web_name_x':'web_name','team_id_x':'team_id','opp_team_id_y':'opp_team_id','was_home_y':'was_home'})

import streamlit as st

# Remove print statement that was causing TypeError
# print("Columns of latest2 before creating final_tbl:")
# print(latest2.columns)

# Create a temporary DataFrame with the required columns, including the correctly named element_type
temp_tbl = latest2[['web_name_x','team_id_x','opp_team_id_y','was_home_y','pred_next_points', 'element_type_y']].copy()

# Rename the columns in the temporary DataFrame
temp_tbl.rename(columns={
    'web_name_x': 'web_name',
    'team_id_x': 'team_id',
    'opp_team_id_y': 'opp_team_id',
    'was_home_y': 'was_home',
    'element_type_y': 'position' # Rename element_type_y to position
}, inplace=True)

# Sort the temporary DataFrame and assign to final_tbl
final_tbl = temp_tbl.sort_values('pred_next_points', ascending=False).copy()


# Map team IDs → short names
TEAM_MAP = teams.set_index("team_id")["short_name"].to_dict()
final_tbl['team'] = final_tbl['team_id'].map(TEAM_MAP)
final_tbl['opp']  = final_tbl['opp_team_id'].map(TEAM_MAP)

# Map positions
POS_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}
final_tbl['position'] = final_tbl['position'].map(POS_MAP) # Now 'position' column exists

# --- Streamlit filters ---
st.sidebar.header("Filters")
pos_choice  = st.sidebar.selectbox("Position", ["All"] + list(POS_MAP.values()))
team_choice = st.sidebar.selectbox("Team", ["All"] + sorted(set(TEAM_MAP.values())))
min_points  = st.sidebar.slider("Minimum predicted points", 0.0, 15.0, 4.0, 0.5)
search      = st.sidebar.text_input("Search player name")

# Apply filters
tbl = final_tbl.copy()
if pos_choice != "All":
    tbl = tbl[tbl['position'] == pos_choice]
if team_choice != "All":
    tbl = tbl[tbl['team'] == team_choice]
tbl = tbl[tbl['pred_next_points'] >= min_points]
if search.strip():
    s = search.strip().lower()
    tbl = tbl[tbl['web_name'].str.lower().str.contains(s)]

st.subheader("Predicted Fantasy Points (Top Players)")
top_n = st.sidebar.slider("How many players to display", 10, 200, 50, 10) # Keep only one definition
st.dataframe(tbl.head(top_n), width='stretch') # Use width='stretch' instead of use_container_width=True

st.download_button(
    label="Download predictions as CSV",
    data=final_tbl.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv"
)

