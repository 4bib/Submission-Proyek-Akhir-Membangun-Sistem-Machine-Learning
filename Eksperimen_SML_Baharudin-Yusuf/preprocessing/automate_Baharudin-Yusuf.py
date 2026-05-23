import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def automate_preprocessing(input_path, output_dir):
    df = pd.read_csv(input_path)

    # Feature Engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    # Outlier Handling
    df_filtered = df[df['median_house_value'] < 500001].copy()
    df_filtered = df_filtered[df_filtered['rooms_per_household'] < 20]
    df_filtered = df_filtered[df_filtered['population_per_household'] < 10]
    df_filtered = df_filtered[df_filtered['bedrooms_per_room'] < 1]

    X = df_filtered.drop('median_house_value', axis=1)
    y = df_filtered['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_test_scaled.npy'), X_test_scaled)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    return "Preprocessing complete. Files saved to " + output_dir

if __name__ == '__main__':
    # Paths relative to the script's location in the repo
    input_csv = '/content/sample_data/california_housing_train.csv'
    out_path = './namadataset_preprocessing'
    automate_preprocessing(input_csv, out_path)
