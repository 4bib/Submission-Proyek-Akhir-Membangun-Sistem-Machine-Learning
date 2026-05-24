"""
Machine Learning Modelling Module
Nama: Baharudin Yusuf
Deskripsi: Training model machine learning menggunakan MLflow Tracking
Dataset: preprocessed_data.csv dari tahap preprocessing
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib


class MLFlowModelling:
    """
    Kelas untuk training dan tracking model machine learning dengan MLflow
    """
    
    def __init__(self, preprocessed_data_path, output_path="./mlflow_models"):
        """
        Initialize MLFlow Modelling
        
        Args:
            preprocessed_data_path (str): Path ke preprocessed dataset
            output_path (str): Path untuk menyimpan model
        """
        self.preprocessed_data_path = preprocessed_data_path
        self.output_path = output_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.metrics = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("Submission-ML-Experiment")
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
    
    def load_preprocessed_data(self):
        """
        Load preprocessed data dari hasil preprocessing
        """
        print("[STEP 1] Loading Preprocessed Data...")
        
        try:
            # Cari file CSV di folder preprocessing
            if os.path.isfile(self.preprocessed_data_path):
                self.df = pd.read_csv(self.preprocessed_data_path)
                print(f"✓ Data loaded from {self.preprocessed_data_path}")
            elif os.path.isdir(self.preprocessed_data_path):
                # Cari file CSV di folder
                csv_files = [f for f in os.listdir(self.preprocessed_data_path) if f.endswith('.csv')]
                if csv_files:
                    file_path = os.path.join(self.preprocessed_data_path, csv_files[0])
                    self.df = pd.read_csv(file_path)
                    print(f"✓ Data loaded from {file_path}")
                else:
                    print("⚠ No CSV files found in directory")
                    return False
            
            print(f"Data shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def prepare_data(self, target_column=None, test_size=0.2, random_state=42):
        """
        Prepare data untuk training
        
        Args:
            target_column (str): Nama kolom target (jika None, ambil kolom terakhir)
            test_size (float): Proporsi test set
            random_state (int): Random seed
        """
        print("\n[STEP 2] Preparing Data for Training...")
        
        if self.df is None:
            print("⚠ No data loaded")
            return False
        
        try:
            # Tentukan target column
            if target_column is None:
                target_column = self.df.columns[-1]
                print(f"Target column not specified, using last column: {target_column}")
            
            if target_column not in self.df.columns:
                print(f"✗ Target column '{target_column}' not found")
                return False
            
            # Split features dan target
            X = self.df.drop(columns=[target_column])
            y = self.df[target_column]
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"✓ Data prepared successfully")
            print(f"  - Training set: {self.X_train.shape}")
            print(f"  - Test set: {self.X_test.shape}")
            return True
            
        except Exception as e:
            print(f"✗ Error preparing data: {str(e)}")
            return False
    
    def train_classification_model(self, model_name="RandomForest", run_name="classification_run"):
        """
        Train classification model dengan MLflow tracking
        
        Args:
            model_name (str): Nama model (RandomForest, LogisticRegression)
            run_name (str): Nama MLflow run
        """
        print(f"\n[STEP 3] Training {model_name} Classification Model...")
        
        if self.X_train is None:
            print("⚠ Data not prepared")
            return False
        
        try:
            with mlflow.start_run(run_name=run_name):
                # Enable autolog
                mlflow.sklearn.autolog()
                
                # Select model
                if model_name == "RandomForest":
                    self.model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    params = {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "model_type": "RandomForestClassifier"
                    }
                elif model_name == "LogisticRegression":
                    self.model = LogisticRegression(random_state=42, max_iter=1000)
                    params = {
                        "max_iter": 1000,
                        "model_type": "LogisticRegression"
                    }
                else:
                    print(f"✗ Model {model_name} not supported")
                    return False
                
                # Log parameters
                mlflow.log_params(params)
                
                # Train model
                print(f"Training {model_name}...")
                self.model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred_train = self.model.predict(self.X_train)
                y_pred_test = self.model.predict(self.X_test)
                
                # Calculate metrics
                self.metrics = {
                    "train_accuracy": accuracy_score(self.y_train, y_pred_train),
                    "test_accuracy": accuracy_score(self.y_test, y_pred_test),
                    "train_precision": precision_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
                    "test_precision": precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
                    "train_recall": recall_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
                    "test_recall": recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
                    "train_f1": f1_score(self.y_train, y_pred_train, average='weighted', zero_division=0),
                    "test_f1": f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0)
                }
                
                # Log metrics
                mlflow.log_metrics(self.metrics)
                
                # Log model signature
                signature = infer_signature(self.X_train, y_pred_train)
                mlflow.sklearn.log_model(self.model, "model", signature=signature)
                
                print(f"✓ Model trained successfully")
                print(f"  - Train Accuracy: {self.metrics['train_accuracy']:.4f}")
                print(f"  - Test Accuracy: {self.metrics['test_accuracy']:.4f}")
                
                return True
                
        except Exception as e:
            print(f"✗ Error training model: {str(e)}")
            return False
    
    def train_regression_model(self, model_name="RandomForest", run_name="regression_run"):
        """
        Train regression model dengan MLflow tracking
        
        Args:
            model_name (str): Nama model (RandomForest, LinearRegression)
            run_name (str): Nama MLflow run
        """
        print(f"\n[STEP 3] Training {model_name} Regression Model...")
        
        if self.X_train is None:
            print("⚠ Data not prepared")
            return False
        
        try:
            with mlflow.start_run(run_name=run_name):
                # Enable autolog
                mlflow.sklearn.autolog()
                
                # Select model
                if model_name == "RandomForest":
                    self.model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    params = {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "model_type": "RandomForestRegressor"
                    }
                elif model_name == "LinearRegression":
                    self.model = LinearRegression()
                    params = {
                        "model_type": "LinearRegression"
                    }
                else:
                    print(f"✗ Model {model_name} not supported")
                    return False
                
                # Log parameters
                mlflow.log_params(params)
                
                # Train model
                print(f"Training {model_name}...")
                self.model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred_train = self.model.predict(self.X_train)
                y_pred_test = self.model.predict(self.X_test)
                
                # Calculate metrics
                self.metrics = {
                    "train_mse": mean_squared_error(self.y_train, y_pred_train),
                    "test_mse": mean_squared_error(self.y_test, y_pred_test),
                    "train_mae": mean_absolute_error(self.y_train, y_pred_train),
                    "test_mae": mean_absolute_error(self.y_test, y_pred_test),
                    "train_r2": r2_score(self.y_train, y_pred_train),
                    "test_r2": r2_score(self.y_test, y_pred_test)
                }
                
                # Log metrics
                mlflow.log_metrics(self.metrics)
                
                # Log model signature
                signature = infer_signature(self.X_train, y_pred_train)
                mlflow.sklearn.log_model(self.model, "model", signature=signature)
                
                print(f"✓ Model trained successfully")
                print(f"  - Train R²: {self.metrics['train_r2']:.4f}")
                print(f"  - Test R²: {self.metrics['test_r2']:.4f}")
                
                return True
                
        except Exception as e:
            print(f"✗ Error training model: {str(e)}")
            return False
    
    def save_model(self, model_path=None):
        """
        Save model ke local storage
        """
        print("\n[STEP 4] Saving Model...")
        
        if self.model is None:
            print("⚠ No model to save")
            return False
        
        try:
            if model_path is None:
                model_path = os.path.join(self.output_path, 'trained_model.pkl')
            
            joblib.dump(self.model, model_path)
            print(f"✓ Model saved to {model_path}")
            
            # Save metrics
            metrics_path = os.path.join(self.output_path, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"✓ Metrics saved to {metrics_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
            return False
    
    def print_summary(self):
        """
        Print training summary
        """
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model: {type(self.model).__name__}")
        print(f"Training Set Size: {self.X_train.shape}")
        print(f"Test Set Size: {self.X_test.shape}")
        print(f"\nMetrics:")
        for metric, value in self.metrics.items():
            print(f"  - {metric}: {value:.4f}")
        print("="*60)


def main():
    """
    Main function untuk training model
    """
    print("="*60)
    print("MACHINE LEARNING MODELLING")
    print("="*60)
    
    # Initialize MLFlow Modelling
    modelling = MLFlowModelling(
        preprocessed_data_path="./namadataset_preprocessing/preprocessed_data.csv",
        output_path="./mlflow_models"
    )
    
    # Load data
    if not modelling.load_preprocessed_data():
        return False
    
    # Prepare data
    if not modelling.prepare_data(target_column=None, test_size=0.2):
        return False
    
    # Train model - untuk classification
    # Ubah ke train_regression_model() jika dataset untuk regression
    if not modelling.train_classification_model(
        model_name="RandomForest",
        run_name="RF-Classification-Basic"
    ):
        return False
    
    # Save model
    if not modelling.save_model():
        return False
    
    # Print summary
    modelling.print_summary()
    
    print("\n✓ MODELLING COMPLETED SUCCESSFULLY")
    print("\nMLflow Tracking UI:")
    print("  - Run: mlflow ui --backend-store-uri ./mlruns")
    print("  - Open: http://localhost:5000")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
