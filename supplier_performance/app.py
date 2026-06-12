
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL  = os.path.join(BASE, 'model')

# ── Load artifacts once at startup ───────────────────────────────────────────
model     = joblib.load(os.path.join(MODEL, 'best_model.pkl'))
scaler    = joblib.load(os.path.join(MODEL, 'scaler.pkl'))
ohe_enc   = joblib.load(os.path.join(MODEL, 'ohe_encoder.pkl'))
od_enc    = joblib.load(os.path.join(MODEL, 'od_encoder.pkl'))

with open(os.path.join(MODEL, 'metrics.json')) as f:
    metrics = json.load(f)

BEST_MODEL  = metrics.pop('_best')
FEAT_COLS   = metrics.pop('_features')

# Dropdown options (must match training data categories)
OHE_COLS     = ['SupplierLocationCode', 'SupplierCategory', 'Supplier_Distance', 'IsSupplierListedComp']
OD_COLS      = ['SupplierValueCategory']
LOCATION_OPTS = ['LOC_A', 'LOC_B', 'LOC_C', 'LOC_D', 'LOC_E']
CATEGORY_OPTS = ['Electronics', 'Raw Materials', 'Packaging', 'Logistics', 'Services']
DISTANCE_OPTS = ['Near', 'Medium', 'Far']
LISTED_OPTS   = ['Yes', 'No']
VALUE_OPTS    = ['Bronze', 'Silver', 'Gold', 'Platinum']


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess_input(form):
    """Transform raw form data into scaled feature array."""
    # Numeric features (order must match training)
    num_vals = {
        'FinancialHealthScore':       float(form['FinancialHealthScore']),
        'SupplierAssociationYears':   float(form['SupplierAssociationYears']),
        'QualityScore':               float(form['QualityScore']),
        'DeliveryScore':              float(form['DeliveryScore']),
        'ComplianceScore':            float(form['ComplianceScore']),
        'PriceCompetitivenessScore':  float(form['PriceCompetitivenessScore']),
        'ResponseTimeHours':          float(form['ResponseTimeHours']),
        'DefectRate':                 float(form['DefectRate']),
        'OnTimeDeliveryRate':         float(form['OnTimeDeliveryRate']),
    }
    num_df = pd.DataFrame([num_vals])

    # OHE categorical features
    ohe_df = pd.DataFrame([[
        form['SupplierLocationCode'],
        form['SupplierCategory'],
        form['Supplier_Distance'],
        form['IsSupplierListedComp'],
    ]], columns=OHE_COLS)
    ohe_arr = ohe_enc.transform(ohe_df)
    ohe_out = pd.DataFrame(ohe_arr, columns=ohe_enc.get_feature_names_out())

    # Ordinal categorical feature
    od_df = pd.DataFrame([[form['SupplierValueCategory']]], columns=OD_COLS)
    od_arr = od_enc.transform(od_df)
    od_out = pd.DataFrame(od_arr, columns=od_enc.get_feature_names_out())

    # Combine → scale
    combined = pd.concat([num_df.reset_index(drop=True),
                          od_out.reset_index(drop=True),
                          ohe_out.reset_index(drop=True)], axis=1)
    # Align columns exactly as training
    combined = combined.reindex(columns=FEAT_COLS, fill_value=0)
    scaled = scaler.transform(combined)
    return scaled


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Home page with prediction form."""
    context = {
        'location_opts': LOCATION_OPTS,
        'category_opts': CATEGORY_OPTS,
        'distance_opts': DISTANCE_OPTS,
        'listed_opts':   LISTED_OPTS,
        'value_opts':    VALUE_OPTS,
        'best_model':    BEST_MODEL,
    }
    return render_template('index.html', **context)


@app.route('/predict', methods=['POST'])
def predict():
    """Process form submission and return prediction result."""
    try:
        features = preprocess_input(request.form)
        pred      = model.predict(features)[0]
        proba     = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)
        label     = 'Good Performer ✅' if pred == 1 else 'Poor Performer ❌'
        label_raw = 'Good' if pred == 1 else 'Poor'

        form_data = dict(request.form)

        return render_template(
            'result.html',
            prediction=pred,
            label=label,
            label_raw=label_raw,
            confidence=confidence,
            proba_good=round(proba[1]*100, 2),
            proba_poor=round(proba[0]*100, 2),
            form_data=form_data,
            best_model=BEST_MODEL,
        )
    except Exception as e:
        return render_template('error.html', error=str(e)), 400


@app.route('/visuals')
def visuals():
    """Charts & visualizations page."""
    return render_template('visuals.html', best_model=BEST_MODEL)


@app.route('/metrics')
def model_metrics():
    """Model evaluation metrics page."""
    return render_template('metrics.html', metrics=metrics, best_model=BEST_MODEL)


@app.route('/about')
def about():
    """About the project."""
    return render_template('about.html')


@app.route('/api/metrics')
def api_metrics():
    """JSON endpoint for metrics (AJAX)."""
    return jsonify(metrics)


@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='Page not found (404)'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='Internal server error (500)'), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
