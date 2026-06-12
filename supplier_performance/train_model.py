"""
Supplier Performance ML Training Script
Trains multiple classifiers, saves best model + charts
"""

import os
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, 'dataset', 'SupplierPerformance.xlsx')
MODEL  = os.path.join(BASE, 'model')
CHARTS = os.path.join(BASE, 'static', 'images')

os.makedirs(MODEL,  exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = ['#4361ee', '#3a0ca3', '#7209b7', '#f72585', '#4cc9f0']
sns.set_theme(style='darkgrid', palette=PALETTE)
plt.rcParams.update({'figure.dpi': 110, 'font.family': 'DejaVu Sans'})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    df = pd.read_excel(DATA)

    # Fill missing values
    df['FinancialHealthScore']   = df['FinancialHealthScore'].fillna(df['FinancialHealthScore'].mean())
    df['SupplierAssociationYears'] = df['SupplierAssociationYears'].fillna(df['SupplierAssociationYears'].mean())
    df['SupplierLocationCode']   = df['SupplierLocationCode'].fillna(df['SupplierLocationCode'].mode()[0])

    # Drop identifier column if present
    if 'SupplierID' in df.columns:
        df = df.drop('SupplierID', axis=1)

    X = df.drop('PerformInd', axis=1)
    y = df['PerformInd']

    # Numeric columns
    num = X.select_dtypes(include=['int64', 'float64'])

    # Categorical columns
    cat = X.select_dtypes(include=['object'])

    ohe_cols = ['SupplierLocationCode', 'SupplierCategory', 'Supplier_Distance', 'IsSupplierListedComp']
    od_cols  = ['SupplierValueCategory']

    ohe_data = cat[ohe_cols]
    od_data  = cat[od_cols]

    ohe_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    od_enc  = OrdinalEncoder()

    ohe_arr = ohe_enc.fit_transform(ohe_data)
    od_arr  = od_enc.fit_transform(od_data)

    new_ohe = pd.DataFrame(ohe_arr, columns=ohe_enc.get_feature_names_out(), index=num.index)
    new_od  = pd.DataFrame(od_arr,  columns=od_enc.get_feature_names_out(),  index=num.index)

    final = pd.concat([num.reset_index(drop=True),
                       new_od.reset_index(drop=True),
                       new_ohe.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    final_scaled = scaler.fit_transform(final)
    final_df = pd.DataFrame(final_scaled, columns=final.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        final_df, y, test_size=0.2, random_state=123, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, ohe_enc, od_enc, final.columns.tolist(), df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Train Models
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression':  LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree':        DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        cv    = cross_val_score(mdl, X_train, y_train, cv=5, scoring='accuracy')
        mse   = mean_squared_error(y_test, preds)

        results[name] = {
            'model':      mdl,
            'preds':      preds,
            'accuracy':   accuracy_score(y_test, preds),
            'r2':         r2_score(y_test, preds),
            'mae':        mean_absolute_error(y_test, preds),
            'mse':        mse,
            'rmse':       np.sqrt(mse),
            'cv_mean':    cv.mean(),
            'cv_std':     cv.std(),
            'cm':         confusion_matrix(y_test, preds),
            'report':     classification_report(y_test, preds, output_dict=True),
        }
        print(f"  {name}: acc={results[name]['accuracy']:.4f}  cv={cv.mean():.4f}±{cv.std():.4f}")

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  Best model → {best_name}")
    return results, best_name


# ─────────────────────────────────────────────────────────────────────────────
# 3. Charts
# ─────────────────────────────────────────────────────────────────────────────
def make_charts(df, results, best_name, X_test, y_test):

    # 3a. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    num_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'heatmap.png'), bbox_inches='tight')
    plt.close()

    # 3b. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = df['PerformInd'].value_counts()
    bars = ax.bar(['Poor (0)', 'Good (1)'], vals.values, color=PALETTE[:2], edgecolor='white', linewidth=1.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3,
                str(int(b.get_height())), ha='center', va='bottom', fontweight='bold')
    ax.set_title('Supplier Performance Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'target_dist.png'), bbox_inches='tight')
    plt.close()

    # 3c. Model Comparison Bar Chart
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    cvs   = [results[n]['cv_mean']  for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, accs, w, label='Test Accuracy', color=PALETTE[0], alpha=0.9)
    ax.bar(x + w/2, cvs,  w, label='CV Mean Acc',   color=PALETTE[2], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'model_comparison.png'), bbox_inches='tight')
    plt.close()

    # 3d. Confusion Matrix (best model)
    cm  = results[best_name]['cm']
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Poor', 'Good'], yticklabels=['Poor', 'Good'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix – {best_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # 3e. Feature Importance (Random Forest)
    rf = results['Random Forest']['model']
    feat_names = X_test.columns.tolist()
    importances = pd.Series(rf.feature_importances_, index=feat_names).nlargest(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.sort_values().plot.barh(ax=ax, color=PALETTE[3], edgecolor='white')
    ax.set_title('Top 12 Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'feature_importance.png'), bbox_inches='tight')
    plt.close()

    # 3f. Numeric distributions
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['PerformInd'])
    num_cols = num_cols[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue='PerformInd', kde=True, ax=axes[i],
                     palette=[PALETTE[0], PALETTE[3]], alpha=0.7)
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')
    plt.suptitle('Feature Distributions by Performance', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'distributions.png'), bbox_inches='tight')
    plt.close()

    # 3g. CV Score per model
    fig, ax = plt.subplots(figsize=(8, 4))
    cv_means = [results[n]['cv_mean'] for n in names]
    cv_stds  = [results[n]['cv_std']  for n in names]
    ax.bar(names, cv_means, yerr=cv_stds, color=PALETTE[1], alpha=0.85,
           capsize=6, edgecolor='white', linewidth=1.2)
    ax.set_ylabel('CV Accuracy')
    ax.set_title('5-Fold Cross Validation Scores', fontsize=12, fontweight='bold')
    ax.set_xticklabels(names, rotation=12, ha='right')
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS, 'cv_scores.png'), bbox_inches='tight')
    plt.close()

    print("  Charts saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save Model Artifacts
# ─────────────────────────────────────────────────────────────────────────────
def save_artifacts(results, best_name, scaler, ohe_enc, od_enc, feature_cols):
    best_mdl = results[best_name]['model']
    joblib.dump(best_mdl, os.path.join(MODEL, 'best_model.pkl'))
    joblib.dump(scaler,   os.path.join(MODEL, 'scaler.pkl'))
    joblib.dump(ohe_enc,  os.path.join(MODEL, 'ohe_encoder.pkl'))
    joblib.dump(od_enc,   os.path.join(MODEL, 'od_encoder.pkl'))

    metrics = {}
    for name, r in results.items():
        metrics[name] = {
            'accuracy':  round(r['accuracy'], 4),
            'r2':        round(r['r2'], 4),
            'mae':       round(r['mae'], 4),
            'mse':       round(r['mse'], 4),
            'rmse':      round(r['rmse'], 4),
            'cv_mean':   round(r['cv_mean'], 4),
            'cv_std':    round(r['cv_std'], 4),
            'report':    r['report'],
        }
    metrics['_best'] = best_name
    metrics['_features'] = feature_cols

    with open(os.path.join(MODEL, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Artifacts saved → model/ (best: {best_name})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n[1] Loading & preprocessing data …")
    X_train, X_test, y_train, y_test, scaler, ohe_enc, od_enc, feat_cols, df = load_and_preprocess()

    print("[2] Training models …")
    results, best_name = train_models(X_train, X_test, y_train, y_test)

    print("[3] Generating charts …")
    make_charts(df, results, best_name, X_test, y_test)

    print("[4] Saving artifacts …")
    save_artifacts(results, best_name, scaler, ohe_enc, od_enc, feat_cols)

    print("\n✅  Training complete!\n")
