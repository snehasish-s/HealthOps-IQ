# ============================================================
#  WEEK 2 - STEP 8 : ML MODEL — 30-Day Readmission Prediction
#  Project : HealthOps IQ — Team 1 (Clinical Analytics)
#  Models  : Logistic Regression + Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)

os.makedirs("ml_outputs", exist_ok=True)

# ── STEP 1 : Load cleaned data ──────────────────────────────
df = pd.read_csv("D:\\HealthOps IQ\\diabetic_data_cleaned.csv")
df.replace("?", np.nan, inplace=True)
df = df[df["gender"].isin(["Male", "Female"])]

# ── STEP 2 : Define binary target (1 = readmitted <30 days) ─
df["readmitted_flag"] = (df["readmitted"] == "<30").astype(int)

# ── STEP 3 : Feature engineering (mirror Silver layer) ──────
# Age midpoint
age_map = {"[0-10)":5,"[10-20)":15,"[20-30)":25,"[30-40)":35,
           "[40-50)":45,"[50-60)":55,"[60-70)":65,"[70-80)":75,
           "[80-90)":85,"[90-100)":95}
df["age_numeric"] = df["age"].map(age_map).fillna(0)

# Total prior visits
df["total_prior_visits"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)

# Encode medication columns ordinally (No=0, Steady=1, Down=2, Up=3)
med_cols = ["metformin","repaglinide","nateglinide","chlorpropamide",
            "glimepiride","glipizide","glyburide","pioglitazone",
            "rosiglitazone","acarbose","insulin"]
med_order = {"No":0,"Steady":1,"Down":2,"Up":3}
for m in med_cols:
    if m in df.columns:
        df[m + "_enc"] = df[m].map(med_order).fillna(0)

# Active medication count
df["active_medication_count"] = sum(
    (df[m + "_enc"] > 0).astype(int) for m in med_cols if m + "_enc" in df.columns
)

# Binary indicators
df["change_encoded"]        = (df["change"] == "Ch").astype(int)
df["diabetes_med_encoded"]  = (df["diabetesMed"] == "Yes").astype(int)
df["gender_encoded"]        = (df["gender"] == "Female").astype(int)
df["is_diabetic_primary"]   = df["diag_1"].astype(str).str.startswith("250").astype(int)
df["high_risk_patient"]     = (df["number_inpatient"] > 0).astype(int)
df["long_stay"]             = (df["time_in_hospital"] > 7).astype(int)

# ── STEP 4 : Select feature columns ─────────────────────────
feature_cols = [
    "age_numeric", "gender_encoded",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "total_prior_visits", "active_medication_count",
    "change_encoded", "diabetes_med_encoded",
    "is_diabetic_primary", "high_risk_patient", "long_stay",
] + [m + "_enc" for m in med_cols if m + "_enc" in df.columns]

X = df[feature_cols].fillna(0)
y = df["readmitted_flag"]

print(f"Features: {len(feature_cols)} | Rows: {len(X):,}")
print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

# ── STEP 5 : Train/test split (stratified) ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── STEP 6 : Model 1 — Logistic Regression ──────────────────
print("\n=== Logistic Regression ===")
logreg = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)
logreg.fit(X_train_s, y_train)
lr_pred  = logreg.predict(X_test_s)
lr_proba = logreg.predict_proba(X_test_s)[:, 1]

print(classification_report(y_test, lr_pred, digits=3))
print(f"AUC-ROC: {roc_auc_score(y_test, lr_proba):.3f}")

# ── STEP 7 : Model 2 — Random Forest ────────────────────────
print("\n=== Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=20,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)          # trees don't need scaling
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, rf_pred, digits=3))
print(f"AUC-ROC: {roc_auc_score(y_test, rf_proba):.3f}")

# ── STEP 8 : Confusion matrix + ROC plots ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d",
            cmap="Blues", ax=axes[0])
axes[0].set_title("Random Forest — Confusion Matrix")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

for name, proba in [("LogReg", lr_proba), ("RandomForest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curve"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].legend()
plt.tight_layout()
plt.savefig("ml_outputs/roc_confusion.png", dpi=150)

# ── STEP 9 : Feature importance (Random Forest) ─────────────
importances = pd.Series(rf.feature_importances_, index=feature_cols) \
                .sort_values(ascending=False)
print("\nTop 10 features:\n", importances.head(10))

plt.figure(figsize=(9, 6))
importances.head(15).plot(kind="barh", color="#3498db")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("ml_outputs/feature_importance.png", dpi=150)

# ── STEP 10 : Risk score 0–100 per patient ──────────────────
df_scores = X.copy()
df_scores["encounter_id"]   = df["encounter_id"].values
df_scores["actual"]         = y.values
df_scores["risk_score"]     = (rf.predict_proba(X)[:, 1] * 100).round(1)
df_scores["risk_band"]      = pd.cut(
    df_scores["risk_score"], bins=[-1, 30, 60, 100],
    labels=["Low", "Medium", "High"]
)
risk_out = df_scores[["encounter_id", "actual", "risk_score", "risk_band"]]
risk_out.to_csv("ml_outputs/patient_risk_scores.csv", index=False)
print("\nRisk band distribution:\n", risk_out["risk_band"].value_counts())

# ── STEP 11 : Save model + scaler ───────────────────────────
joblib.dump(rf, "ml_outputs/readmission_rf_model.pkl")
joblib.dump(scaler, "ml_outputs/scaler.pkl")
joblib.dump(feature_cols, "ml_outputs/feature_cols.pkl")
print("\n✔ Model, scaler, and risk scores saved to ml_outputs/")
