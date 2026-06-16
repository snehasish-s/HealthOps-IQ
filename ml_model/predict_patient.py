# ============================================================
#  WEEK 3 - STEP 10 : LIVE DEMO — Patient Readmission Risk
#  Project : HealthOps IQ — Team 1 (Clinical Analytics)
# ============================================================

import joblib
import numpy as np
import pandas as pd

rf           = joblib.load("ml_outputs/readmission_rf_model.pkl")
feature_cols = joblib.load("ml_outputs/feature_cols.pkl")

AGE_MAP = {"[0-10)":5,"[10-20)":15,"[20-30)":25,"[30-40)":35,
           "[40-50)":45,"[50-60)":55,"[60-70)":65,"[70-80)":75,
           "[80-90)":85,"[90-100)":95}
MED_ORDER = {"No":0, "Steady":1, "Down":2, "Up":3}
MED_COLS  = ["metformin","repaglinide","nateglinide","chlorpropamide",
             "glimepiride","glipizide","glyburide","pioglitazone",
             "rosiglitazone","acarbose","insulin"]

# ── Population averages → used to explain WHY a patient is high-risk ──
POP_MEANS = {
    "number_inpatient": 0.6, "total_prior_visits": 1.4,
    "num_medications": 16, "time_in_hospital": 4.4,
    "num_lab_procedures": 43, "number_diagnoses": 7.4,
    "high_risk_patient": 0.34, "num_procedures": 1.3,
}


def build_features(patient: dict) -> pd.DataFrame:
    f = {}
    f["age_numeric"]    = AGE_MAP.get(patient.get("age"), 0)
    f["gender_encoded"] = 1 if patient.get("gender") == "Female" else 0
    f["time_in_hospital"]   = patient.get("time_in_hospital", 0)
    f["num_lab_procedures"] = patient.get("num_lab_procedures", 0)
    f["num_procedures"]     = patient.get("num_procedures", 0)
    f["num_medications"]    = patient.get("num_medications", 0)
    f["number_outpatient"]  = patient.get("number_outpatient", 0)
    f["number_emergency"]   = patient.get("number_emergency", 0)
    f["number_inpatient"]   = patient.get("number_inpatient", 0)
    f["number_diagnoses"]   = patient.get("number_diagnoses", 0)
    f["total_prior_visits"] = (
        f["number_outpatient"] + f["number_emergency"] + f["number_inpatient"]
    )
    f["change_encoded"]       = 1 if patient.get("change") == "Ch" else 0
    f["diabetes_med_encoded"] = 1 if patient.get("diabetesMed") == "Yes" else 0
    f["is_diabetic_primary"]  = 1 if str(patient.get("diag_1", "")).startswith("250") else 0
    f["high_risk_patient"]    = 1 if f["number_inpatient"] > 0 else 0
    f["long_stay"]            = 1 if f["time_in_hospital"] > 7 else 0

    meds = patient.get("medications", {})
    active = 0
    for m in MED_COLS:
        val = MED_ORDER.get(meds.get(m, "No"), 0)
        f[m + "_enc"] = val
        if val > 0:
            active += 1
    f["active_medication_count"] = active

    row = {c: f.get(c, 0) for c in feature_cols}
    return pd.DataFrame([row])[feature_cols]


def top_drivers(X):
    """Features where this patient is most elevated above the population average."""
    imp = pd.Series(rf.feature_importances_, index=feature_cols)
    drivers = {}
    for f, mean in POP_MEANS.items():
        if f in X.columns and mean > 0:
            drivers[f] = ((X.iloc[0][f] - mean) / mean) * imp.get(f, 0)
    return pd.Series(drivers).sort_values(ascending=False).head(4).index.tolist()


def score_patient(patient: dict) -> dict:
    X = build_features(patient)
    proba = rf.predict_proba(X)[0, 1]
    score = round(proba * 100, 1)
    band = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    return {"risk_score": score, "risk_band": band,
            "top_factors": top_drivers(X)}          # ← now uses top_drivers


if __name__ == "__main__":
    high_risk = {
        "age": "[70-80)", "gender": "Female",
        "time_in_hospital": 12, "num_lab_procedures": 60,
        "num_procedures": 3, "num_medications": 22,
        "number_outpatient": 2, "number_emergency": 4,
        "number_inpatient": 5, "number_diagnoses": 9,
        "change": "Ch", "diabetesMed": "Yes", "diag_1": "250.6",
        "medications": {"insulin": "Up", "metformin": "Steady"},
    }
    low_risk = {
        "age": "[20-30)", "gender": "Male",
        "time_in_hospital": 2, "num_lab_procedures": 15,
        "num_procedures": 0, "num_medications": 5,
        "number_outpatient": 0, "number_emergency": 0,
        "number_inpatient": 0, "number_diagnoses": 3,
        "change": "No", "diabetesMed": "No", "diag_1": "401",
        "medications": {"insulin": "No"},
    }

    for label, p in [("HIGH-RISK PATIENT", high_risk),
                     ("LOW-RISK PATIENT", low_risk)]:
        r = score_patient(p)
        print("\n" + "=" * 50)
        print(f"  {label}")
        print("=" * 50)
        print(f"  Risk Score : {r['risk_score']}/100")
        print(f"  Risk Band  : {r['risk_band']}")
        print(f"  Top Factors: {', '.join(r['top_factors'])}")
