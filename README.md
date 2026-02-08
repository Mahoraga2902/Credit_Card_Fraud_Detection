# Credit Card Fraud Detection - Mini Project

## 🎯 Overview

A machine learning project that detects fraudulent credit card transactions using a **Random Forest classifier** trained on PCA-transformed features. The project includes:

- **✅ Data Pipeline**: Load and preprocess 284,807 transactions
- **✅ ML Model**: Random Forest trained on 29 features (V1-V28 PCA + Amount)
- **✅ REST API**: Flask backend with fraud detection endpoints
- **✅ Web Frontend**: Beautiful UI for making predictions
- **✅ Performance**: 99.93% accuracy, 81.57% fraud detection rate

---

## 📁 Project Structure

```
creditcardfraud_detection/
├── main.py                          ⭐ MAIN ENTRY POINT - Train the model
├── data_loader.py                   Data loading & preprocessing functions
├── model.py                         Model training & prediction functions
├── app.py                           Flask REST API + Web Frontend
├── test_project.py                  Verification & testing script
├── creditcard.csv                   Original dataset (284,807 transactions)
├── fraud_detection_model.pkl        Pre-trained Random Forest model
├── scaler.pkl                       StandardScaler (for feature normalization)
├── templates1/                      Web UI templates (for future customization)
├── README.md                        This file
├── QUICK_START.md                   Quick reference guide
├── PROJECT_RESTORED.md              Restoration & refactoring details
├── INDEX.md                         Documentation index
└── venv/                            Python virtual environment
```

---

## 🚀 Quick Start

### Option 1: Web Interface (Easiest)
```bash
python app.py
```
Then open browser: `http://localhost:5000/`

Features:
- 🎨 Beautiful purple gradient design
- 📝 Enter 29 comma-separated features
- 🚀 "Load Sample" button for quick testing
- ✓ Real-time predictions with confidence scores
- 🎯 Color-coded results (red = fraud, green = normal)

### Option 2: Train the Model
```bash
python main.py
```

This will:
1. Load creditcard.csv (284,807 transactions)
2. Preprocess: Select 29 features (V1-V28 + Amount)
3. Split: 80% training, 20% testing
4. Scale: StandardScaler normalization
5. Train: Random Forest with 100 trees
6. Evaluate: Show Accuracy, Precision, Recall, F1-Score
7. Save: Model + Scaler files

### Option 3: Test Everything
```bash
python test_project.py
```

Verifies:
- ✅ All files exist
- ✅ All libraries installed
- ✅ Model can load
- ✅ Predictions work
- ✅ API is functional

---

## 🌐 Web Frontend Screenshot

```
╔════════════════════════════════════════════════╗
║   🔍 Credit Card Fraud Detection              ║
║   ML-powered fraud detection system            ║
║════════════════════════════════════════════════╣
║                                                ║
║  Features (comma-separated, 29 values)         ║
║  ┌────────────────────────────────────────┐   ║
║  │ -1.36, -0.07, 2.54, 1.38, ...         │   ║
║  └────────────────────────────────────────┘   ║
║                                                ║
║  [Make Prediction] [Load Sample]               ║
║                                                ║
║  ✓ TRANSACTION NORMAL                         ║
║  Prediction: normal                            ║
║  Confidence: Normal 1.00% | Fraud 0.00%       ║
║                                                ║
╚════════════════════════════════════════════════╝
```

---

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve web frontend |
| `/health` | GET | Check API status |
| `/info` | GET | Get API information |
| `/predict` | POST | Make fraud prediction |

### Example Prediction (PowerShell)

```powershell
$body = @{
    features = @(-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 50.0)
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "normal",
  "confidence": {
    "normal": 1.0,
    "fraud": 0.0
  },
  "fraud_probability": 0.0
}
```

---

## 🎓 How It Works

### Data Flow
```
creditcard.csv (284,807 rows)
    ↓
data_loader.py - Extract 29 features
    ↓
Split 80/20 - StandardScaler normalization
    ↓
model.py - Random Forest training
    ↓
fraud_detection_model.pkl + scaler.pkl
    ↓
app.py - REST API + Web Frontend
    ↓
Predictions (Normal / Fraud)
```

### Model Performance (Test Set: 56,962 transactions)

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.93% |
| **Precision** | 89.67% |
| **Recall** | 81.57% |
| **F1-Score** | 85.37% |

**Interpretation:**
- Correctly classifies 99.93% of all transactions
- When flagging fraud, 89.67% are actually fraudulent
- Catches 81.57% of real fraud cases
- Excellent balance between precision and recall

---

## 📥 Input Format

Provide exactly **29 numeric values** in this order:

```
V1, V2, V3, ..., V28, Amount
```

### Example Values:

**Normal Transaction:**
```
-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 50.0
```

**Fraudulent Transaction:**
```
-2.31, 1.95, -1.61, 4.00, -0.52, -1.43, -2.54, 1.39, -2.77, -2.77, 3.20, -2.90, -0.60, -4.29, 0.39, -1.14, -2.83, -0.02, 0.42, 0.13, 0.52, -0.04, -0.47, 0.32, 0.04, 0.18, 0.26, -0.14, 0.00
```

---

## 📚 File Descriptions

### **main.py**
- Orchestrates the complete ML pipeline
- Loads → Preprocesses → Trains → Evaluates → Saves
- Trains model in ~2 minutes

```bash
python main.py          # Train and evaluate
python main.py --app    # Train, then start Flask API
```

### **data_loader.py**
Functions for data processing:
- `load_data()` - Load CSV
- `preprocess_data()` - Extract features
- `split_and_scale_data()` - Train/test split + scaling
- `prepare_data()` - Complete pipeline

### **model.py**
Functions for ML:
- `train_model()` - Train Random Forest
- `evaluate_model()` - Calculate metrics
- `save_model()` - Save to disk
- `load_model()` - Load from disk
- `predict_transaction()` - Make predictions

### **app.py**
Flask REST API:
- Serves web frontend (HTML/CSS/JavaScript)
- Handles `/predict` requests
- Provides `/health` and `/info` endpoints
- Integrates model.py for predictions

---

## 🔧 Requirements

**Python 3.8+** with libraries:
- pandas
- numpy
- scikit-learn
- joblib
- flask
- flask-cors

**Install:**
```bash
pip install pandas numpy scikit-learn joblib flask flask-cors
```

---

## 📊 Dataset Information

**creditcard.csv**
- **Rows**: 284,807 transactions
- **Columns**: 31 (Time, V1-V28, Amount, Class)
- **Class Distribution**:
  - Normal: 284,314 (99.83%)
  - Fraudulent: 493 (0.17%)
- **Class Imbalance**: 578:1 ratio

---

## 🛠️ Programmatic Usage

```python
from model import load_model, predict_transaction
import numpy as np

# Load model and scaler
model, scaler = load_model()

# Create feature vector (29 values)
features = np.array([
    -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
    0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
    0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
    0.13, -0.19, 0.13, -0.02, 50.0
])

# Make prediction
prediction, probabilities = predict_transaction(model, scaler, features)

print(f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
print(f"Normal: {probabilities[0]:.2%}, Fraud: {probabilities[1]:.2%}")
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model files not found" | Run `python main.py` to train |
| "CSV not found" | Ensure creditcard.csv exists |
| "Library missing" | `pip install scikit-learn flask` |
| "Port 5000 busy" | Change port in app.py line: `app.run(port=5001)` |
| "28 features instead of 29" | Check you have all V1-V28 + Amount |
| "API won't start" | Verify model files exist in directory |

---

## 📋 Features

- ✅ **Ready to Use** - Pre-trained model included
- ✅ **Web Interface** - Beautiful UI for predictions
- ✅ **REST API** - JSON endpoints for integration
- ✅ **Modular Code** - Easy to modify and extend
- ✅ **Comprehensive Documentation** - Multiple guides included
- ✅ **Production Ready** - Professional error handling and logging
- ✅ **Well Tested** - Verification script included
- ✅ **Clean Project** - Only essential files kept

---

## 🔮 Future Enhancements

- SMOTE for class imbalance
- Hyperparameter tuning
- K-fold cross-validation
- Feature importance analysis
- Model versioning
- Automated retraining pipeline

---

## 📜 License

Uses **Kaggle Credit Card Fraud Detection Dataset**
- Educational/research purposes
- Attribution required for redistribution

---

**Status**: ✅ **Production Ready**  
**Last Updated**: February 7, 2026  
**Accuracy**: 99.93% | **Fraud Detection**: 81.57%
