
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_tuned_model():
    # Load processed data
    data_path = './namadataset_preprocessing'
    X_train = np.load(os.path.join(data_path, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_scaled.npy'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).values.ravel()

    mlflow.set_experiment("California_Housing_Project")

    # Manual logging context
    with mlflow.start_run(run_name="Tuned_RandomForest"):
        rf = RandomForestRegressor(random_state=42)

        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }

        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Manual Logging (Skilled Criteria requirement)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(best_model, "tuned_model")

        print(f"Tuned Model trained. Best Params: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

if __name__ == '__main__':
    train_tuned_model()
