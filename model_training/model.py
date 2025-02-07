import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

# Load dataset from CSV
# Replace 'path_to_your_file.csv' with the actual path to your CSV file
data = pd.read_csv('data/iris.csv')

# Assuming your CSV has a 'target' column for labels, and the rest are features
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# 1. Hyperparameter Tuning:
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Start an MLflow run to track this experiment
with mlflow.start_run():
    # Perform GridSearchCV for Hyperparameter Tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Log parameters and the best accuracy score in MLflow
    mlflow.log_params(best_params)  # Log the hyperparameters
    mlflow.log_metric("accuracy", grid_search.best_score_)  # Log the best score

    # Documenting results
    print("Best Parameters:", best_params)
    print("Best Accuracy:", grid_search.best_score_)

    # 2. Model Packaging
    # Create an input example for the model to define the input signature
    input_example = X_train.iloc[:1].to_dict(orient="records")[0]  # Taking the first row as an input example

    # Log the trained model with MLflow, passing the input_example to define the signature
    mlflow.sklearn.log_model(best_model, "model", input_example=input_example)

    # Optionally, save the best model using pickle (for example purposes)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
