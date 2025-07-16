# Import libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def load_data():
    """Load and save dataset"""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    
    # Save raw data
    df.to_csv('data/raw_california_housing.csv', index=False)
    print("✅ Data saved to data/raw_california_housing.csv")
    return df

def preprocess_data(df):
    """Split and scale data"""
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✅ Scaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    """Train and save Random Forest model"""
    model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/housing_predictor.pkl')
    print("✅ Model saved to models/housing_predictor.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    """Calculate performance metrics"""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📊 Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return rmse, r2

if __name__ == "__main__":
    # Execute pipeline
    print("🚀 Starting California Housing Prediction Pipeline...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print("✅ Pipeline completed successfully!")
