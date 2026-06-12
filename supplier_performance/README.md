# Supplier Performance Prediction — Flask ML App

A complete machine learning web application that predicts whether a supplier will be
a **Good** or **Poor Performer** using classification models trained on supplier attributes.

---

## 📁 Project Structure

```
supplier_performance/
├── app.py                    # Flask application (main entry point)
├── train_model.py            # ML training script
├── requirements.txt          # Python dependencies
│
├── dataset/
│   └── SupplierPerformance.xlsx   # Dataset
│
├── model/
│   ├── best_model.pkl        # Trained best model
│   ├── scaler.pkl            # StandardScaler
│   ├── ohe_encoder.pkl       # OneHotEncoder
│   ├── od_encoder.pkl        # OrdinalEncoder
│   └── metrics.json          # All evaluation metrics
│
├── templates/
│   ├── base.html             # Shared layout + navbar
│   ├── index.html            # Prediction form (home)
│   ├── result.html           # Prediction result
│   ├── visuals.html          # Charts & visualizations
│   ├── metrics.html          # Model evaluation metrics
│   ├── about.html            # About the project
│   └── error.html            # Error page
│
└── static/
    └── images/               # Generated chart PNGs
        ├── heatmap.png
        ├── target_dist.png
        ├── model_comparison.png
        ├── confusion_matrix.png
        ├── feature_importance.png
        ├── distributions.png
        └── cv_scores.png
```

---

## ⚙️ Installation & Setup

### 1. Clone / Download the project
```bash
cd supplier_performance
```

### 2. Create & activate a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train_model.py
```
This will:
- Load and preprocess the dataset
- Train 4 ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Select the best model by accuracy
- Generate 7 charts saved to `static/images/`
- Save model artifacts to `model/`

### 5. Run the Flask app
```bash
python app.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 🤖 ML Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline, fast, interpretable |
| Decision Tree | Non-linear, max_depth=6 |
| Random Forest | Ensemble, 150 estimators |
| Gradient Boosting | Ensemble, 100 estimators |

---

## 📊 Evaluation Metrics

- Accuracy
- R² Score
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- 5-Fold Cross-Validation Score
- Confusion Matrix
- Full Classification Report (Precision, Recall, F1)

---

## 🌐 Routes

| Route | Description |
|---|---|
| `GET /` | Home — prediction form |
| `POST /predict` | Submit form → get prediction result |
| `GET /visuals` | All charts and visualizations |
| `GET /metrics` | Model evaluation metrics |
| `GET /about` | About the project |
| `GET /api/metrics` | JSON API endpoint for metrics |
