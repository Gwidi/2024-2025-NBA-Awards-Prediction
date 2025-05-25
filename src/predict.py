import mlflow
import mlflow.pyfunc
import pandas as pd

# Load the dataset with player data
df = pd.read_csv('../data/league_leaders_2024-25.csv')
X_test = features = df.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM'])
print(X_test.head())


# Load the model back for predictions as a generic Python Function model
# Replace 'model_uri' with the actual path or URI to your MLflow model
model_uri = "../mlartifacts/524645951732338284/bfd2812fe4f24f198ee64a5a17093541/artifacts/nba_model" 
loaded_model = mlflow.pyfunc.load_model(model_uri)

predictions = loaded_model.predict(X_test)



result = pd.DataFrame({
    "PLAYER": df["PLAYER"],
    "predicted_class": predictions
})

result.to_csv('../data/predictions_all_nba_teams.csv', index=False)