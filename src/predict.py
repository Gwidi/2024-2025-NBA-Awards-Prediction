import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with player data
df = pd.read_csv('../data/league_leaders_2024-25.csv')
df = df[(df['GP'] >= 65)]
X_latest = features = df.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM'])
#X_latest = X_latest[['EFF','PTS','FTA', 'FGM','FTM','FGA','TOV','MIN','DREB']]



# Load the model back for predictions as a generic Python Function model
# Replace 'model_uri' with the actual path or URI to your MLflow model
model_uri = "../mlartifacts/524645951732338284/1cbf3797b16847d4aa50c4d039d7392a/artifacts/nba_model" 
loaded_model = mlflow.sklearn.load_model(model_uri)

# Załóżmy, że używałeś tych kolumn:
feature_names = X_latest.columns

# Ważności cech
importances = loaded_model.feature_importances_

# Stwórz DataFrame
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Wykres słupkowy
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'][:15][::-1], feature_importance_df['importance'][:15][::-1])
plt.xlabel("Ważność cechy")
plt.title("Top 15 najważniejszych cech (RandomForestClassifier)")
plt.tight_layout()
plt.show()


# Predict probabilities
probs = loaded_model.predict_proba(X_latest)[:, 1]

# Dodaj je do ramki
latest_season = df.copy()
latest_season['all_nba_prob'] = probs

print(latest_season[['PLAYER', 'all_nba_prob']])

# Sortuj i wybierz Top 15
top15 = latest_season.sort_values(by='all_nba_prob', ascending=False).head(15)

# Przypisz teamy
top15['pred_team'] = [1]*5 + [2]*5 + [3]*5

# Pokaż wynik
print(top15[['PLAYER', 'all_nba_prob', 'pred_team']])


# result = pd.DataFrame({
#     "PLAYER": df["PLAYER"],
#     "predicted_class": predictions
# })

# result.to_csv('../data/predictions_all_nba_teams.csv', index=False)

# # Function to generate JSON file with NBA teams
# def generate_all_nba_json(first_team, second_team, third_team, first_rookie_team, second_rookie_team, filename):
#     data = {
#         "first all-nba team": first_team,
#         "second all-nba team": second_team,
#         "third all-nba team": third_team,
#         "first rookie all-nba team": first_rookie_team,
#         "second rookie all-nba team": second_rookie_team
#     }

#     with open(filename, 'w') as json_file:
#         json.dump(data, json_file, indent=2)