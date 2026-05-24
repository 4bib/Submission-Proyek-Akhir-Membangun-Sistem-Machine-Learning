
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_model():
    # Load processed data
    data_path = './Membangun_model/namadataset_preprocessing'
    X_train = np.load(os.path.join(data_path, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test_scaled.npy'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    mlflow.set_experiment("California_Housing_Project")

    # Enable autologging for Basic criteria
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Production_Model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Final Model trained. RMSE: {np.sqrt(mse):.2f}, R2: {r2:.4f}")

if __name__ == '__main__':
    train_model()
