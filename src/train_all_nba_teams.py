import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Load the dataset
data = pd.read_csv('../data/final_dataset_all_nba_teams.csv')

# Choose the features and labels
features = data.drop(columns=['PLAYER_ID', 'RANK', 'PLAYER', 'TEAM_ID', 'TEAM', 'SEASON'])
labels = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Log the model parameters and metrics with MLflow
params = {"use_label_encoder": False, "eval_metric": 'logloss'}

# Train the XGBoost model
model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Predykcja i ocena modelu
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Create a new MLflow Experiment
mlflow.set_experiment("Training NBA Teams Model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic XGBoost model for NBA teams classification")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="nba_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
