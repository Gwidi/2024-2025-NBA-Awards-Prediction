import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import GridSearchCV
import joblib


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Load the dataset
data = pd.read_csv('../data/league_leaders_2000_2024_with_rookies_and_all_rookie_team.csv', index_col=False)


data = data[(data['GP'] >= 65)]
data = data[(data['IS_ROOKIE'] == True)]

# Podział na sezony treningowe i testowe
train_seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2000, 2018)]  # 2000-01 to 2019-20
test_seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2018, 2024)]   # 2020-21 to 2023-24

train_data = data[data["SEASON"].isin(train_seasons)]
test_data = data[data["SEASON"].isin(test_seasons)]


# Prepare the features and labels
X_train = train_data.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'SEASON', 'Tm','IS_ROOKIE', 'IS_ALL_ROOKIE_TEAM'])
y_train = train_data['IS_ALL_ROOKIE_TEAM']
X_test = test_data.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'SEASON', 'Tm','IS_ROOKIE', 'IS_ALL_ROOKIE_TEAM'])
y_test = test_data['IS_ALL_ROOKIE_TEAM']

X_train = X_train[['EFF', 'PTS', 'FTA', 'FGM', 'FTM', 'FGA', 'TOV', 'MIN', 'DREB']]
X_test = X_test[['EFF', 'PTS', 'FTA', 'FGM', 'FTM', 'FGA', 'TOV', 'MIN', 'DREB']]


# Enode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)


# Log the model parameters and metrics with MLflow
params = {'class_weight': 'balanced', 'n_estimators': 100, 'random_state': 42}

# Train the RandomForestClassifier model
model = RandomForestClassifier(**params)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_weighted',  # lub 'f1'
    cv=5,
    n_jobs=-1,
    verbose=1
)




grid_search.fit(X_train, y_train_encoded)
best_model = grid_search.best_estimator_

# Save the best model to file
joblib.dump(best_model, "../models/nba_rookie_model.pkl")

# Prediction and evaluation of the model
y_pred = best_model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
print(classification_report(y_test, y_pred_labels))
print(f"Accuracy: {accuracy_score(y_test, y_pred_labels)}")

# Create a new MLflow Experiment
mlflow.set_experiment("Training NBA Rookie Teams Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Log the loss metric
    mlflow.log_metric("f1_score", grid_search.best_score_)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Limites the number of parameters")
    


    dataset_path = "../data/league_leaders_2000_2024_with_rookies_and_all_rookie_team.csv"
    mlflow.log_artifact(dataset_path, artifact_path="dataset")

    # Infer the model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model = best_model,
        artifact_path="nba_rookie_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="RandomForestClassifier",
    )
