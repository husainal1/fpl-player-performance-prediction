# FPL Player Performance 

This project uses machine learning (XGBoost regression) to predict upcoming Fantasy Premier League (FPL) player performance. 
The model is trained on official FPL match-by-match data (via the public FPL API) with feature engineering such as:
- Player form (lagged states, rolling averages for goals, assists, and points)
- Team strength metrics (attack vs defense, home vs away)
- Availability indicators (minutes playes, recent match participation)
- Match context (opponent strength, fixture difficulty, gameweek timing)

Key Features 
- Automated data ingestion from the [FPl API] ((https://fantasy.premierleague.com/api).
- XGBoost regressor to forecast each player's expected fantasy points in the next gameweek.
- Evalutaion with baseline compareison (3-game rolling average vs ML model).
- Custom filters for insights (e.g. best underdog players outside the Big 6).
- Outpur predictions in ranked tables and exportable CSV.

Project Structure
- 'fpl_predictor.py' -> main pipeline: data collection, feature engineering, model training, predictions.

