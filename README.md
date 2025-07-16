# California Housing Price Prediction 🏠

Predict median house values in California districts using machine learning.

## Project Overview
- **Goal**: Predict housing prices based on district attributes
- **Algorithm**: Random Forest Regressor
- **Performance**: 
  - RMSE: ~0.52 
  - R²: ~0.80

## Repository Structure
```
.
├── data/                   # Raw and processed data
├── models/                 # Trained models and scalers
├── src/                    # Source code
│   └── main.py             # Main processing script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Installation
```bash
git clone https://github.com/yourusername/california-housing-prediction.git
cd california-housing-prediction
pip install -r requirements.txt
```

### Run Pipeline
```bash
python src/main.py
```

### Expected Output
```
🚀 Starting California Housing Prediction Pipeline...
✅ Data saved to data/raw_california_housing.csv
✅ Scaler saved to models/scaler.pkl
✅ Model saved to models/housing_predictor.pkl
📊 Model Evaluation:
RMSE: 0.5172
R² Score: 0.8023
✅ Pipeline completed successfully!
```

## Make Predictions
```python
import joblib
import numpy as np

# Load artifacts
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/housing_predictor.pkl')

# Sample data (8 features)
sample = np.array([8.3252, 41.0, 6.984, 1.024, 322.0, 2.55, 37.88, -122.23]).reshape(1, -1)

# Preprocess and predict
scaled_sample = scaler.transform(sample)
prediction = model.predict(scaled_sample)

print(f"🏠 Predicted House Value: ${prediction[0]*100000:.2f}")
```

## License
MIT License - Free for academic and commercial use
