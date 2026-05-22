"""
Automated Preprocessing Module
Nama: Baharudin Yusuf
Deskripsi: Fungsi untuk melakukan preprocessing secara otomatis
Proses yang sama dengan notebook eksperimen, tetapi dalam struktur yang berbeda
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime


class DataPreprocessor:
    """
    Kelas untuk mengotomatisasi proses preprocessing data
    """
    
    def __init__(self, raw_data_path, output_path):
        """
        Initialize preprocessor
        
        Args:
            raw_data_path (str): Path ke folder raw dataset
            output_path (str): Path untuk menyimpan hasil preprocessing
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.df = None
        self.original_shape = None
        self.processed_shape = None
        
        # Buat output directory jika belum ada
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """
        Step 1: Load data dari raw dataset
        """
        print("[STEP 1] Loading Data...")
        try:
            # Cek file yang tersedia
            raw_files = os.listdir(self.raw_data_path)
            print(f"Files found in {self.raw_data_path}: {raw_files}")
            
            # Cari file CSV, Excel, atau JSON
            for file in raw_files:
                if file.endswith('.csv'):
                    file_path = os.path.join(self.raw_data_path, file)
                    self.df = pd.read_csv(file_path)
                    print(f"✓ Data loaded from {file}")
                    break
                elif file.endswith(('.xlsx', '.xls')):
                    file_path = os.path.join(self.raw_data_path, file)
                    self.df = pd.read_excel(file_path)
                    print(f"✓ Data loaded from {file}")
                    break
                elif file.endswith('.json'):
                    file_path = os.path.join(self.raw_data_path, file)
                    self.df = pd.read_json(file_path)
                    print(f"✓ Data loaded from {file}")
                    break
            
            if self.df is None:
                print("⚠ No compatible data files found. Creating sample dataframe...")
                self.df = pd.DataFrame()
            
            self.original_shape = self.df.shape
            print(f"Data shape: {self.original_shape}")
            return self.df
            
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def exploratory_data_analysis(self):
        """
        Step 2: Exploratory Data Analysis (EDA)
        """
        print("\n[STEP 2] Exploratory Data Analysis...")
        
        if self.df is None or self.df.empty:
            print("⚠ No data to analyze")
            return
        
        print(f"\nDataset Info:")
        print(f"  - Shape: {self.df.shape}")
        print(f"  - Columns: {list(self.df.columns)}")
        print(f"  - Data types:\n{self.df.dtypes}")
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  - No missing values")
        
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        
        print(f"\nDuplicate Rows: {self.df.duplicated().sum()}")
        print("✓ EDA completed")
    
    def handle_missing_values(self):
        """
        Step 3a: Handle missing values
        """
        print("\n[STEP 3a] Handling Missing Values...")
        
        if self.df is None or self.df.empty:
            return
        
        missing_before = self.df.isnull().sum().sum()
        
        # Strategi: Drop rows dengan missing values
        self.df = self.df.dropna()
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  - Missing values before: {missing_before}")
        print(f"  - Missing values after: {missing_after}")
        print("✓ Missing values handled")
    
    def remove_duplicates(self):
        """
        Step 3b: Remove duplicate rows
        """
        print("\n[STEP 3b] Removing Duplicates...")
        
        if self.df is None or self.df.empty:
            return
        
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_after = self.df.duplicated().sum()
        
        print(f"  - Duplicates before: {duplicates_before}")
        print(f"  - Duplicates after: {duplicates_after}")
        print("✓ Duplicates removed")
    
    def data_type_conversion(self):
        """
        Step 3c: Convert data types
        """
        print("\n[STEP 3c] Data Type Conversion...")
        
        if self.df is None or self.df.empty:
            return
        
        # Identifikasi kolom numerik dan kategori
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Coba konversi ke numerik jika memungkinkan
                try:
                    self.df[col] = pd.to_numeric(self.df[col])
                    print(f"  - Converted {col} to numeric")
                except:
                    pass
        
        print("✓ Data types optimized")
    
    def feature_engineering(self):
        """
        Step 4: Feature Engineering (opsional sesuai kebutuhan)
        """
        print("\n[STEP 4] Feature Engineering...")
        
        if self.df is None or self.df.empty:
            return
        
        # Contoh: Tambahkan fitur baru jika diperlukan
        # self.df['new_feature'] = self.df['col1'] * self.df['col2']
        
        print("✓ Feature engineering completed")
    
    def normalization_scaling(self):
        """
        Step 5: Normalization/Scaling (untuk kolom numerik)
        """
        print("\n[STEP 5] Normalization/Scaling...")
        
        if self.df is None or self.df.empty:
            return
        
        from sklearn.preprocessing import StandardScaler
        
        # Identifikasi kolom numerik
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            print(f"  - Scaled {len(numeric_cols)} numeric columns")
        
        print("✓ Normalization/Scaling completed")
    
    def save_preprocessed_data(self):
        """
        Step 6: Save preprocessed data
        """
        print("\n[STEP 6] Saving Preprocessed Data...")
        
        if self.df is None or self.df.empty:
            print("⚠ No data to save")
            return
        
        # Save sebagai CSV
        output_file = os.path.join(self.output_path, 'preprocessed_data.csv')
        self.df.to_csv(output_file, index=False)
        print(f"✓ Data saved to {output_file}")
        
        # Save metadata
        self.processed_shape = self.df.shape
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': self.original_shape,
            'processed_shape': self.processed_shape,
            'columns': list(self.df.columns),
            'dtypes': str(self.df.dtypes.to_dict())
        }
        
        import json
        metadata_file = os.path.join(self.output_path, 'preprocessing_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to {metadata_file}")
    
    def preprocess(self):
        """
        Main preprocessing pipeline - menjalankan semua steps
        """
        print("="*60)
        print("AUTOMATED PREPROCESSING PIPELINE")
        print("="*60)
        
        try:
            # Jalankan semua steps
            self.load_data()
            self.exploratory_data_analysis()
            self.handle_missing_values()
            self.remove_duplicates()
            self.data_type_conversion()
            self.feature_engineering()
            self.normalization_scaling()
            self.save_preprocessed_data()
            
            print("\n" + "="*60)
            print("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Original shape: {self.original_shape}")
            print(f"Processed shape: {self.processed_shape}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ ERROR during preprocessing: {str(e)}")
            return False


def main():
    """
    Main function untuk menjalankan preprocessing
    """
    # Tentukan paths
    raw_data_path = "namadataset_raw"
    output_path = "preprocessing/namadataset_preprocessing"
    
    # Jalankan preprocessing
    preprocessor = DataPreprocessor(raw_data_path, output_path)
    success = preprocessor.preprocess()
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
