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

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Load the dataset
data = pd.read_csv('../data/final_dataset_all_nba_teams.csv', index_col=False)

# Create a new binary column if the player is in the All-NBA team
data['all_nba'] = data['Tm'].apply(lambda x: 0 if x == '0' else 1)

data = data[(data['GP'] >= 65)]


# Choose the features and labels
features = data.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'SEASON', 'Tm', 'all_nba'])
#features = features[['EFF','PTS','FTA', 'FGM','FTM','FGA','TOV','MIN','DREB']]
labels = data['all_nba']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.3, random_state=42, stratify=labels)


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
    scoring='f1',  # lub 'f1'
    cv=5,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predykcja i ocena modelu
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Create a new MLflow Experiment
mlflow.set_experiment("Training NBA Teams Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Log the loss metric
    mlflow.log_metric("f1_score", grid_search.best_score_)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Changed number of parameters")
    


    dataset_path = "../data/final_dataset_all_nba_teams.csv"
    mlflow.log_artifact(dataset_path, artifact_path="dataset")

    # Infer the model signature
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model = best_model,
        artifact_path="nba_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="RandomForestClassifier",
    )
