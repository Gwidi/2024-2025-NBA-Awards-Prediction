import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import joblib
import argparse

parser = argparse.ArgumentParser(description="Predict All-NBA teams and save to JSON.")
parser.add_argument("--output", type=str, default="../predictions/gwidon_szczepankiewicz.json", help="Ścieżka do pliku wyjściowego JSON")
args = parser.parse_args()


# Load the dataset with player data
df = pd.read_csv('../data/final_dataset_for_prediction.csv', index_col=False)
df_all_nba = df[(df['GP'] >= 65) & (df['IS_ROOKIE'] == False)]
X_latest = features = df_all_nba.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'IS_ROOKIE'])
X_latest = X_latest[['EFF','PTS','FTA', 'FGM','FTM','FGA','TOV','MIN','DREB']]



# Load the model back for predictions as a generic Python Function model
# Replace 'model_uri' with the actual path or URI to your MLflow model
# model_uri = "../mlartifacts/524645951732338284/952cf48f112a486684fef9eddfe871c6/artifacts/nba_model" 
# loaded_model = mlflow.sklearn.load_model(model_uri)
loaded_model = joblib.load('../models/nba_model.pkl')

# # Załóżmy, że używałeś tych kolumn:
# feature_names = X_latest.columns

# # Ważności cech
# importances = loaded_model.feature_importances_

# # Stwórz DataFrame
# feature_importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': importances
# }).sort_values(by='importance', ascending=False)

# # Wykres słupkowy
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['feature'][:15][::-1], feature_importance_df['importance'][:15][::-1])
# plt.xlabel("Ważność cechy")
# plt.title("Top 15 najważniejszych cech (RandomForestClassifier)")
# plt.tight_layout()
# plt.show()



# Predict 
results = loaded_model.predict(X_latest)

# Kodowanie etykiet
label_encoder = LabelEncoder()
#y_pred_labels = label_encoder.inverse_transform(results)

# Dodaj je do ramki
latest_season = df_all_nba.copy()
latest_season['all_nba_results'] = results

# Sort the results by team order (1st>2nd>3rd>0)
team_order = {'1': 1, '2': 2, '3': 3, '0': 4}
latest_season['team_order'] = latest_season['all_nba_results'].map(team_order)

# Sort the DataFrame by EFF 
latest_season = latest_season.sort_values(['team_order', 'PTS'], ascending=[True, False])

# Choose the top players for each team
first_team = []
second_team = []
third_team = []

used_indices = set()

for team_list, team_name in zip([first_team, second_team, third_team], ['1', '2', '3']):
    # Choose players for the current team 
    candidates = latest_season[(latest_season['all_nba_results'] == team_name) & (~latest_season.index.isin(used_indices))]
    # I there's too few candidates, fill with players from the next teams
    needed = 5 - len(candidates)
    selected = candidates.head(5)
    team_list.extend(selected['PLAYER'].tolist())
    used_indices.update(selected.index)
    if needed > 0:
        # If there are not enough candidates, take from the next team
        extra_candidates = latest_season[~latest_season.index.isin(used_indices)]
        extra_selected = extra_candidates.head(needed)
        team_list.extend(extra_selected['PLAYER'].tolist())
        used_indices.update(extra_selected.index)

print("First team:", first_team)
print("Second team:", second_team)
print("Third team:", third_team)
                                               ### PREDICTING ROOKIES ###
# model_uri_rookie = "../mlartifacts/637011297769987143/18402d7b557f4ab79299f5e8cbe7ec25/artifacts/nba_rookie_model"
# loaded_model_rookie = mlflow.sklearn.load_model(model_uri_rookie)
loaded_model_rookie = joblib.load('../models/nba_rookie_model.pkl')

df_rookies = df[(df['GP'] >= 65) & (df['IS_ROOKIE'] == True)]

features_rookie = df_rookies.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'IS_ROOKIE'])
features_rookie = features_rookie[['EFF','PTS','FTA', 'FGM','FTM','FGA','TOV','MIN','DREB']]

# Predict 
results_rookie = loaded_model.predict(features_rookie)

# Dodaj je do ramki
latest_season_rookie = df_rookies.copy()
latest_season_rookie['rookie_results'] = results_rookie

# Sort the results by team order (1st>2nd>3rd>0)
team_order = {'1': 1, '2': 2, '0': 3}
latest_season_rookie['team_order'] = latest_season_rookie['rookie_results'].map(team_order)

# Sort the DataFrame by EFF 
latest_season_rookie = latest_season_rookie.sort_values(['team_order', 'PTS'], ascending=[True, False])

# Choose the top players for each team
first_team_rookie = []
second_team_rookie = []

used_indices_rookie = set()

for team_list, team_name in zip([first_team_rookie, second_team_rookie], ['1', '2']):
    # Choose players for the current team 
    candidates = latest_season_rookie[(latest_season_rookie['rookie_results'] == team_name) & (~latest_season_rookie.index.isin(used_indices_rookie))]
    # I there's too few candidates, fill with players from the next teams
    needed = 5 - len(candidates)
    selected = candidates.head(5)
    team_list.extend(selected['PLAYER'].tolist())
    used_indices_rookie.update(selected.index)
    if needed > 0:
        # If there are not enough candidates, take from the next team
        extra_candidates = latest_season_rookie[~latest_season_rookie.index.isin(used_indices_rookie)]
        extra_selected = extra_candidates.head(needed)
        team_list.extend(extra_selected['PLAYER'].tolist())
        used_indices_rookie.update(extra_selected.index)

print("First team Rookie:", first_team_rookie)
print("Second team Rookie:", second_team_rookie)


# Tworzenie słownika w odpowiednim formacie
nba_teams = {
    "first all-nba team": first_team,
    "second all-nba team": second_team,
    "third all-nba team": third_team,
    "first rookie all-nba team": first_team_rookie,
    "second rookie all-nba team": second_team_rookie
}

# Save to JSON file
with open(args.output, "w", encoding="utf-8") as f:
    json.dump(nba_teams, f, indent=2, ensure_ascii=False)