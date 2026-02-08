# Credit Card Fraud Detection - Full-Stack Application

## 🎯 Overview

A production-ready **fraud detection system** with machine learning backend and web frontend featuring user authentication, admin dashboards, and adaptive learning.

**Key Stats:**
- **Model Accuracy**: 99.96% | **Fraud Recall**: 82.83%
- **Data**: 284,807 credit card transactions
- **Architecture**: Flask REST API + HTML/CSS/JS frontend
- **Features**: Authentication, predictions, admin controls, feedback collection, model retraining with safety

---

## ✨ Features

### 🔐 Authentication & Authorization
- User registration with email
- Secure login system
- Session-based authentication
- Admin & regular user roles
- Password validation

### 📊 User Dashboard
- Make fraud predictions on transactions
- View prediction history
- Submit feedback on predictions ("correct" / "incorrect")
- Simple input form: Amount, Transaction Type, Card Present
- Automatic feature mapping to model

### 👨‍💼 Admin Dashboard
- View all users and their predictions
- System statistics (total users, total predictions)
- Feedback collection stats
- Trigger model retraining with safety checks
- Adaptive learning progress tracking

### 🤖 Adaptive Learning
- Collect feedback from predictions
- Track feedback accuracy
- Retrain model using original data + feedback
- Automatic backups before retraining
- Model validation before deployment
- Prevents feedback-only corruption

### 🚀 REST API
- `/predict` - Make fraud predictions (simplified or raw features)
- `/feedback` - Submit prediction feedback
- `/feedback-stats` - Get adaptive learning stats
- `/retrain` - Trigger model retraining (admin only)
- `/health` - API status check

---

## 📁 Project Structure

```
creditcardfraud_detection/
├── app.py                           Flask REST API + Web routes
├── model.py                         ML model & prediction functions
├── adaptive_learning.py              Feedback collection & retraining
├── data_loader.py                   Data preprocessing
├── main.py                          Model training pipeline
│
├── templates1/                      Web UI templates
│   ├── LOGIN-MAIN.html              Login page
│   ├── REGISTER.html                Registration page
│   ├── USER.html                    User dashboard + predictions
│   ├── ADMIN.html                   Admin dashboard
│   └── style.css                    Global styling
│
├── fraud_detection_model.pkl        Trained Random Forest model
├── scaler.pkl                       Feature scaler
├── fraud_detection_model.pkl.backup Model backup (for safety)
│
├── users.json                       User accounts (NOT for GitHub)
├── users.example.json               Template for users.json
├── adaptive_feedback.json            Collected feedback
│
├── requirements.txt                 Python dependencies
├── .gitignore                       Git ignore rules
│
├── README.md                        This file
├── RECOVERY_LOG.md                  Safety mechanisms & fixes
├── QUICK_START.md                   Getting started guide
├── PROJECT_RESTORED.md              Restoration notes
└── INDEX.md                         Documentation index
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```bash
python main.py
```
Creates `fraud_detection_model.pkl` and `scaler.pkl` (takes ~3 min)

### 3. Start the App
```bash
python app.py
```

Then open browser: **`http://localhost:5000`**

### 4. Login
**Admin Account:**
- Username: `admin`
- Password: `admin123`

**Test User:**
- Username: `testuser`
- Password: `password123`

---

## 🎨 How to Use

### Making a Prediction
1. Login to the system
2. Navigate to your dashboard
3. Enter transaction details:
   - **Amount**: Dollar amount
   - **Transaction Type**: Domestic or International
   - **Card Present**: Yes/No
4. Click "Check Transaction"
5. View prediction result with confidence scores

### Providing Feedback
1. After a prediction, you'll see the result
2. Click either:
   - **"This is correct"** - if prediction matches reality
   - **"This is incorrect"** - if prediction was wrong
3. System records feedback for learning

### Admin: Collecting Feedback
1. Login as admin
2. Go to Admin Dashboard
3. See "Adaptive Learning" stats:
   - Total feedback entries
   - Feedback accuracy
   - Ready for retraining (≥10 entries)

### Admin: Retraining the Model
1. Collect 10+ feedback entries from users
2. Go to Admin Dashboard
3. Click "Retrain Model"
4. System will:
   - Create backup of current model
   - Combine original training data + feedback
   - Train new Random Forest
   - Validate the new model
   - Deploy if valid, rollback if not

---

## 🌐 API Endpoints

### Public Endpoints
| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/login` | GET/POST | No | User login |
| `/register` | GET/POST | No | New user registration |
| `/health` | GET | No | API status |

### Authenticated Endpoints
| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/` | GET | Yes | Home/dashboard |
| `/dashboard` | GET | Yes | User dashboard |
| `/predict` | POST | Yes | Make prediction |
| `/feedback` | POST | Yes | Submit feedback |
| `/feedback-stats` | GET | Yes | Get feedback stats |

### Admin-Only Endpoints
| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/admin` | GET | Admin | Admin dashboard |
| `/retrain` | POST | Admin | Trigger retraining |

### Example: Make a Prediction
```powershell
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
# Login
Invoke-WebRequest -Uri "http://localhost:5000/login" -Method POST `
  -Body @{username='admin'; password='admin123'} -WebSession $session -UseBasicParsing | Out-Null

# Make prediction
$payload = @{
  amount = 150.00
  transaction_type = "International"
  card_present = $false
} | ConvertTo-Json

$r = Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method POST `
  -Body $payload -ContentType "application/json" -WebSession $session -UseBasicParsing

$r.Content | ConvertFrom-Json | ConvertTo-Json
```

### Example: Submit Feedback
```powershell
$feedback = @{
  features = @(0) * 29
  prediction = 1
  actual_label = 1
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/feedback" -Method POST `
  -Body $feedback -ContentType "application/json" -WebSession $session -UseBasicParsing
```

---

## 🤖 Model Details

### Training Data
- **Dataset**: Kaggle Credit Card Fraud Detection
- **Samples**: 284,807 transactions
- **Features**: 29 (V1-V28 PCA features + Amount)
- **Class**: Binary (Normal=0, Fraud=1)
- **Split**: 80% train, 20% test

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Trees**: 100 estimators
- **Scaler**: StandardScaler normalization
- **Feature Preprocessing**: PCA-transformed features

### Performance (Test Set)
| Metric | Score |
|--------|-------|
| Accuracy | 99.96% |
| Precision | 85.20% |
| Recall | 82.83% |
| F1-Score | 83.98% |

---

## 🔒 Safety Mechanisms

### Model Corruption Prevention
1. **Mandatory Original Data**: Retraining REQUIRES original training data
   - Feedback-only mode is **disabled**
   - Prevents small bad datasets from corrupting model
   
2. **Automatic Backups**: Before retraining
   - Creates `fraud_detection_model.pkl.backup`
   - Instant rollback if something fails
   
3. **Model Validation**: Before deployment
   - Tests new model can make predictions
   - Aborts retraining if validation fails
   
4. **Explicit Logging**: All retraining events logged
   - Timestamp, samples used, status recorded
   - Audit trail for debugging

### How Adaptive Learning is Safe
```
Original model trained on 284,807 transactions
         ↓
User submits 10-50 feedback examples (high quality)
         ↓
Retrain combines: 284,807 original + 10-50 feedback
         ↓
Result: New model is 99.98% original data + 0.02-0.05% feedback
         ↓
Minimal risk of corruption by bad feedback
```

---

## 📊 File Guide

### Core Application
- **`app.py`** - Flask server, routes, authentication, template rendering
- **`model.py`** - Model loading, prediction, evaluation functions
- **`adaptive_learning.py`** - Feedback collection, model retraining, backup/restore
- **`data_loader.py`** - Data loading and preprocessing

### Training & Setup
- **`main.py`** - Complete ML pipeline (load → train → evaluate → save)
- **`quick_retrain.py`** - Fast model retrain from scratch
- **`requirements.txt`** - All Python dependencies

### Testing & Verification
- **`test_project.py`** - System verification script
- **`test_model.py`** - Model-specific tests
- **`verify_feedback_flow.py`** - End-to-end feedback flow test
- **`verify_recovery.py`** - Model recovery verification

### Frontend
- **`templates1/LOGIN-MAIN.html`** - Login page
- **`templates1/REGISTER.html`** - Registration page
- **`templates1/USER.html`** - User dashboard with predictions & feedback
- **`templates1/ADMIN.html`** - Admin dashboard & controls
- **`templates1/style.css`** - Global styling

### Configuration
- **`users.json`** - User accounts (local, not in GitHub)
- **`users.example.json`** - Template for users.json
- **`.gitignore`** - Git ignore rules
- **`RECOVERY_LOG.md`** - Technical details on safety mechanisms

---

## 🔧 Configuration

### Change Admin Password
Edit `users.json`:
```json
{
  "admin": {
    "password": "your_new_password"
  }
}
```

### Change Flask Port
In `app.py`, change the port in main:
```python
app.run(debug=True, port=5001)  # Change 5000 to your port
```

### Disable Adaptive Learning
In `app.py`, comment out feedback endpoints

---

## 🛠️ Development

### Run Tests
```bash
python test_project.py
python verify_feedback_flow.py
python verify_recovery.py
```

### Retrain Model
```bash
python quick_retrain.py    # Fast retrain from scratch
# or
python main.py             # Full pipeline with evaluation
```

### Debug Mode
```bash
python app.py              # Starts on http://localhost:5000
                           # Debug mode enabled, auto-reload
```

---

## ⚠️ Important Notes

- **Keep `users.json` local** - Never commit to GitHub (has passwords)
- **Model files are large** - `fraud_detection_model.pkl` (~2.7MB) stored locally
- **Backup created before retrain** - Gives safety margin for recovery
- **Feedback needs validation** - 10+ examples recommended for retraining
- **Admin-only retrain** - Prevents accidental model changes

---

## 📝 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Login fails" | Check `users.json` exists with correct credentials |
| "Prediction not working" | Ensure model files (`fraud_detection_model.pkl`, `scaler.pkl`) exist |
| "Port 5000 in use" | Use different port or kill process using port |
| "Can't retrain" | Need ≥10 feedback entries + original training data |
| "Model too conservative" | Feedback helps fine-tune—collect more examples |
| "Git push includes large files" | Check `.gitignore` is correct, use `git status` |

---

## 🔄 Workflow Example

**Day 1: Deploy**
1. `python app.py` - Start server
2. Users login and make predictions
3. Users provide feedback on results

**Day 5: Retrain (10+ feedback collected)**
1. Admin logs in → Admin Dashboard
2. Sees "Ready for retraining" is True
3. Clicks "Retrain Model"
4. System:
   - Backs up current model
   - Trains new model (original + 15 feedback)
   - Validates predictions work
   - Deploys if OK, rolls back if not
5. New model live for next predictions

---

## 📚 More Information

- **RECOVERY_LOG.md** - Details on safety mechanisms and fixes
- **QUICK_START.md** - Fast reference guide
- **PROJECT_RESTORED.md** - History of restoration work
- **INDEX.md** - Documentation index

---

## 🔮 Future Enhancements

- Real-time model performance monitoring
- A/B testing for new models
- Automated retraining schedule
- Model versioning & rollback UI
- Explainability (feature importance)
- Multi-model ensemble
- Custom rule engine for fraud patterns

---

**Status**: ✅ **Production Ready**  
**Last Updated**: February 8, 2026  
**Model**: Random Forest | **Accuracy**: 99.96% | **Fraud Detection**: 82.83%

## ✨ Features

### 🔐 Authentication & Authorization
- User registration with email
- Secure login system
- Session-based authentication
- Admin & regular user roles
- Password validation

### 📊 User Dashboard
- Make fraud predictions on transactions
- View prediction history
- Submit feedback on predictions ("correct" / "incorrect")
- Simple input form: Amount, Transaction Type, Card Present
- Automatic feature mapping to model

### 👨‍💼 Admin Dashboard
- View all users and their predictions
- System statistics (total users, total predictions)
- Feedback collection stats
- Trigger model retraining with safety checks
- Adaptive learning progress tracking

### 🤖 Adaptive Learning
- Collect feedback from predictions
- Track feedback accuracy
- Retrain model using original data + feedback
- Automatic backups before retraining
- Model validation before deployment
- Prevents feedback-only corruption

### 🚀 REST API
- `/predict` - Make fraud predictions (simplified or raw features)
- `/feedback` - Submit prediction feedback
- `/feedback-stats` - Get adaptive learning stats
- `/retrain` - Trigger model retraining (admin only)
- `/health` - API status check

## 📁 Project Structure

```
creditcardfraud_detection/
├── app.py                           Flask REST API + Web routes
├── model.py                         ML model & prediction functions
├── adaptive_learning.py              Feedback collection & retraining
├── data_loader.py                   Data preprocessing
├── main.py                          Model training pipeline
│
├── templates1/                      Web UI templates
│   ├── LOGIN-MAIN.html              Login page
│   ├── REGISTER.html                Registration page
│   ├── USER.html                    User dashboard + predictions
│   ├── ADMIN.html                   Admin dashboard
│   └── style.css                    Global styling
│
├── fraud_detection_model.pkl        Trained Random Forest model
├── scaler.pkl                       Feature scaler
├── fraud_detection_model.pkl.backup Model backup (for safety)
│
├── users.json                       User accounts (NOT for GitHub)
├── users.example.json               Template for users.json
├── adaptive_feedback.json            Collected feedback
│
├── requirements.txt                 Python dependencies
├── .gitignore                       Git ignore rules
│
├── README.md                        This file
├── RECOVERY_LOG.md                  Safety mechanisms & fixes
├── QUICK_START.md                   Getting started guide
├── PROJECT_RESTORED.md              Restoration notes
└── INDEX.md                         Documentation index
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```bash
python main.py
```
Creates `fraud_detection_model.pkl` and `scaler.pkl` (takes ~3 min)

### 3. Start the App
```bash
python app.py
```

Then open browser: **`http://localhost:5000`**

### 4. Login
**Admin Account:**
- Username: `admin`
- Password: `admin123`

**Test User:**
- Username: `testuser`
- Password: `password123`

## 🎨 How to Use

### Making a Prediction
1. Login to the system
2. Navigate to your dashboard
3. Enter transaction details:
   - **Amount**: Dollar amount
   - **Transaction Type**: Domestic or International
   - **Card Present**: Yes/No
4. Click "Check Transaction"
5. View prediction result with confidence scores

### Providing Feedback
1. After a prediction, you'll see the result
2. Click either:
   - **"This is correct"** - if prediction matches reality
   - **"This is incorrect"** - if prediction was wrong
3. System records feedback for learning

### Admin: Collecting Feedback
1. Login as admin
2. Go to Admin Dashboard
3. See "Adaptive Learning" stats:
   - Total feedback entries
   - Feedback accuracy
   - Ready for retraining (≥10 entries)

### Admin: Retraining the Model
1. Collect 10+ feedback entries from users
2. Go to Admin Dashboard
3. Click "Retrain Model"
4. System will:
   - Create backup of current model
   - Combine original training data + feedback
   - Train new Random Forest
   - Validate the new model
   - Deploy if valid, rollback if not

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
