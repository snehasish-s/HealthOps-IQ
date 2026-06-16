# ============================================================
#  WEEK 1 - STEP 3 : Python ETL → Load into MySQL
#  Project  : HealthOps IQ — Team 1 (Clinical Analytics)
#  Dataset  : diabetic_data_cleaned.csv (output from Step 2)
# ============================================================
#
#  HOW TO RUN IN VS CODE:
#  1. Make sure MySQL is running locally
#  2. Run schema_design.sql first in MySQL Workbench
#  3. Open terminal in VS Code (Ctrl + `)
#  4. pip install pandas numpy mysql-connector-python sqlalchemy
#  5. Update DB_CONFIG below with your MySQL password
#  6. python etl_load_mysql.py
# ============================================================

import pandas as pd
import numpy as np
import mysql.connector
from sqlalchemy import create_engine
import time
from urllib.parse import quote_plus

# ─────────────────────────────────────────────────────────────
#  ⚙️  DATABASE CONFIG — UPDATE YOUR PASSWORD HERE
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",          # your MySQL username
    "password": "Snehasish@123",  # ← CHANGE THIS
    "database": "healthops_iq"
}

# ─────────────────────────────────────────────────────────────
#  HELPER : Print progress
# ─────────────────────────────────────────────────────────────
def log(msg):
    print(f"  ✔  {msg}")

def section(title):
    print(f"\n{'='*55}")
    print(f"   {title}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
#  STEP 1 : Load & Clean CSV
# ─────────────────────────────────────────────────────────────
section("STEP 1 : Load Dataset")

df = pd.read_csv("diabetic_data_cleaned.csv")
log(f"Loaded diabetic_data_cleaned.csv → {df.shape[0]:,} rows, {df.shape[1]} cols")

# If running from raw file instead, do cleaning here too
df.replace("?", np.nan, inplace=True)
df = df[df["gender"].isin(["Male", "Female"])]   # drop 3 invalid rows
df.drop_duplicates(subset=["encounter_id"], keep="first", inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"After dedup: {df.shape[0]:,} rows")

# Rename medication columns with hyphens (MySQL doesn't like hyphens)
rename_map = {
    "glyburide-metformin"       : "glyburide_metformin",
    "glipizide-metformin"       : "glipizide_metformin",
    "glimepiride-pioglitazone"  : "glimepiride_pioglitazone",
    "metformin-rosiglitazone"   : "metformin_rosiglitazone",
    "metformin-pioglitazone"    : "metformin_pioglitazone",
}
df.rename(columns=rename_map, inplace=True)
log("Renamed hyphenated medication columns")


# ─────────────────────────────────────────────────────────────
#  STEP 2 : Connect to MySQL
# ─────────────────────────────────────────────────────────────
section("STEP 2 : Connect to MySQL")

try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    log(f"Connected to MySQL → database: {DB_CONFIG['database']}")
except Exception as e:
    print(f"\n  ❌ Connection failed: {e}")
    print("     Check your DB_CONFIG (host, user, password, database)")
    exit(1)

# Also create SQLAlchemy engine for bulk inserts (faster)
encoded_password = quote_plus(DB_CONFIG["password"])

engine = create_engine(
    f"mysql+mysqlconnector://{DB_CONFIG['user']}:{encoded_password}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)
log("SQLAlchemy engine created for bulk loading")


# ─────────────────────────────────────────────────────────────
#  STEP 3 : Load dim_patient
# ─────────────────────────────────────────────────────────────
section("STEP 3 : Loading dim_patient")

dim_patient = (
    df[["patient_nbr", "race", "gender", "age"]]
    .rename(columns={"age": "age_group"})
    .drop_duplicates(subset=["patient_nbr"])
    .copy()
)
dim_patient["race"] = dim_patient["race"].fillna("Unknown")
dim_patient.reset_index(drop=True, inplace=True)

log(f"Unique patients: {len(dim_patient):,}")

# Truncate old data and reload
cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
cursor.execute("TRUNCATE TABLE dim_patient;")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
conn.commit()

dim_patient.to_sql(
    name="dim_patient",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=5000
)
log(f"dim_patient loaded → {len(dim_patient):,} rows")

# Fetch back with auto-generated keys
dim_patient_db = pd.read_sql("SELECT patient_key, patient_nbr FROM dim_patient", engine)
log("Fetched patient_key mapping from DB")


# ─────────────────────────────────────────────────────────────
#  STEP 4 : Load dim_diagnosis
# ─────────────────────────────────────────────────────────────
section("STEP 4 : Loading dim_diagnosis")

med_cols = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone",
    "metformin_rosiglitazone", "metformin_pioglitazone"
]

diag_cols = ["encounter_id", "diag_1", "diag_2", "diag_3", "number_diagnoses"] + med_cols
dim_diagnosis = df[diag_cols].copy()

# Fill nulls for diagnosis codes
dim_diagnosis["diag_1"] = dim_diagnosis["diag_1"].fillna("Unknown")
dim_diagnosis["diag_2"] = dim_diagnosis["diag_2"].fillna("Unknown")
dim_diagnosis["diag_3"] = dim_diagnosis["diag_3"].fillna("Unknown")

# Fill med columns with 'No'
for col in med_cols:
    if col in dim_diagnosis.columns:
        dim_diagnosis[col] = dim_diagnosis[col].fillna("No")

cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
cursor.execute("TRUNCATE TABLE dim_diagnosis;")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
conn.commit()

dim_diagnosis.to_sql(
    name="dim_diagnosis",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=5000
)
log(f"dim_diagnosis loaded → {len(dim_diagnosis):,} rows")

# Fetch back keys
dim_diagnosis_db = pd.read_sql(
    "SELECT diagnosis_key, encounter_id FROM dim_diagnosis", engine
)
log("Fetched diagnosis_key mapping from DB")


# ─────────────────────────────────────────────────────────────
#  STEP 5 : Load dim_admission
# ─────────────────────────────────────────────────────────────
section("STEP 5 : Loading dim_admission")

# Admission type descriptions (from IDS_mapping.csv)
admission_type_map = {
    1: "Emergency", 2: "Urgent", 3: "Elective",
    4: "Newborn",   5: "Not Available", 6: "Not Available",
    7: "Trauma Center", 8: "Not Mapped"
}

admission_source_map = {
    1: "Physician Referral", 2: "Clinic Referral",
    3: "HMO Referral",       4: "Transfer from Hospital",
    5: "Transfer from SNF",  6: "Transfer from Another",
    7: "Emergency Room",     8: "Court/Law Enforcement",
    9: "Not Available",     10: "Transfer from Critical Access",
    11: "Normal Delivery",  13: "Hospice",
    14: "Cortege",          17: "Normal Delivery",
    20: "Not Mapped",       22: "Not Mapped", 25: "Not Mapped"
}

discharge_map = {
    1: "Discharged to Home",
    2: "Transferred to Short-Term Hospital",
    3: "Transferred to SNF",
    4: "Transferred to ICF",
    5: "Transferred to Inpatient Care",
    6: "Home with Home Health Service",
    7: "Left AMA",
    8: "Home under Care of Hospice",
    9: "Admitted as Inpatient",
    10: "Neonate to Another Hospital",
    11: "Expired",
    12: "Still Patient",
    13: "Hospice / Home",
    14: "Hospice / Medical Facility",
    15: "Swing Bed",
    16: "Transferred to Outpatient",
    17: "Transferred to Psychiatric",
    18: "Transferred to Rehab",
    19: "Transferred to Long-Term Care",
    20: "Transferred to Nursing Facility",
    22: "Rehabilitation",
    23: "Long-Term Care",
    24: "Medicaid-Certified Nursing Facility",
    25: "Not Mapped",
    27: "Not Mapped",
    28: "Not Mapped"
}

# Build unique combinations from data
dim_admission = (
    df[["admission_type_id", "discharge_disposition_id", "admission_source_id"]]
    .drop_duplicates()
    .copy()
)
dim_admission["admission_type_desc"]  = dim_admission["admission_type_id"].map(admission_type_map).fillna("Unknown")
dim_admission["discharge_desc"]       = dim_admission["discharge_disposition_id"].map(discharge_map).fillna("Unknown")
dim_admission["admission_source_desc"]= dim_admission["admission_source_id"].map(admission_source_map).fillna("Unknown")
dim_admission.reset_index(drop=True, inplace=True)

cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
cursor.execute("TRUNCATE TABLE dim_admission;")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
conn.commit()

dim_admission.to_sql(
    name="dim_admission",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=1000
)
log(f"dim_admission loaded → {len(dim_admission):,} rows")

# Fetch back keys
dim_admission_db = pd.read_sql(
    "SELECT admission_key, admission_type_id, discharge_disposition_id, admission_source_id FROM dim_admission",
    engine
)
log("Fetched admission_key mapping from DB")


# ─────────────────────────────────────────────────────────────
#  STEP 6 : Load dim_time (synthetic buckets)
# ─────────────────────────────────────────────────────────────
section("STEP 6 : Loading dim_time (synthetic)")

# Dataset spans 1999-2008, no actual dates per encounter
# We create 10 yearly buckets (one per year)
import calendar

time_rows = []
for yr in range(1999, 2009):
    for mo in range(1, 13):
        time_rows.append({
            "year"        : yr,
            "quarter"     : (mo - 1) // 3 + 1,
            "month"       : mo,
            "month_name"  : calendar.month_name[mo],
            "week_of_year": mo * 4,
            "day_of_week" : "Weekday",
            "is_weekend"  : 0
        })

dim_time = pd.DataFrame(time_rows)

cursor.execute("TRUNCATE TABLE dim_time;")
conn.commit()

dim_time.to_sql(
    name="dim_time",
    con=engine,
    if_exists="append",
    index=False
)
log(f"dim_time loaded → {len(dim_time)} rows (1999-2008 monthly buckets)")

# Assign a time_key per row in main df (round-robin across years)
time_db = pd.read_sql("SELECT time_key, year, month FROM dim_time", engine)
# Each encounter gets a time_key based on row position
df["time_key"] = (df.index % len(time_db)) + 1


# ─────────────────────────────────────────────────────────────
#  STEP 7 : Merge keys and load fact_admissions
# ─────────────────────────────────────────────────────────────
section("STEP 7 : Loading fact_admissions")

# Merge patient_key
df = df.merge(dim_patient_db, on="patient_nbr", how="left")
log("Merged patient_key")

# Merge diagnosis_key
df = df.merge(dim_diagnosis_db, on="encounter_id", how="left")
log("Merged diagnosis_key")

# Merge admission_key
df = df.merge(
    dim_admission_db,
    on=["admission_type_id", "discharge_disposition_id", "admission_source_id"],
    how="left"
)
log("Merged admission_key")

# Binary readmission flag: 1 if readmitted within 30 days, else 0
df["readmitted_flag"] = (df["readmitted"] == "<30").astype(int)

# Select only fact table columns
fact_cols = [
    "patient_key", "diagnosis_key", "admission_key", "time_key",
    "encounter_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient",
    "change", "diabetesMed",
    "readmitted", "readmitted_flag"
]

fact_admissions = df[fact_cols].rename(columns={
    "change"     : "medication_change",
    "diabetesMed": "diabetes_med_prescribed"
}).copy()

# Drop rows with missing keys
before = len(fact_admissions)
fact_admissions.dropna(subset=["patient_key", "diagnosis_key", "admission_key"], inplace=True)
fact_admissions[["patient_key","diagnosis_key","admission_key","time_key"]] = \
    fact_admissions[["patient_key","diagnosis_key","admission_key","time_key"]].astype(int)
after = len(fact_admissions)
log(f"Dropped {before - after} rows with missing keys | Final: {after:,} rows")

cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
cursor.execute("TRUNCATE TABLE fact_admissions;")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
conn.commit()

start = time.time()
fact_admissions.to_sql(
    name="fact_admissions",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=5000,
    method="multi"
)
elapsed = round(time.time() - start, 1)
log(f"fact_admissions loaded → {len(fact_admissions):,} rows in {elapsed}s")


# ─────────────────────────────────────────────────────────────
#  STEP 8 : Verify Row Counts
# ─────────────────────────────────────────────────────────────
section("STEP 8 : Verification Queries")

tables = ["dim_patient", "dim_diagnosis", "dim_admission", "dim_time", "fact_admissions"]
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  {table:<25} → {count:>8,} rows")

# Spot-check: readmission breakdown
print()
readmit_check = pd.read_sql(
    "SELECT readmitted, COUNT(*) AS cnt FROM fact_admissions GROUP BY readmitted",
    engine
)
print("  Readmission breakdown in fact table:")
print(readmit_check.to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  STEP 9 : Sample KPI Queries (test your schema)
# ─────────────────────────────────────────────────────────────
section("STEP 9 : Sample KPI Queries")

print("\n  KPI 1 — Readmission Rate by Age Group:")
q1 = """
    SELECT
        p.age_group,
        COUNT(*)                                          AS total_visits,
        SUM(f.readmitted_flag)                            AS readmitted_30d,
        ROUND(SUM(f.readmitted_flag) * 100.0 / COUNT(*), 2) AS readmit_rate_pct
    FROM fact_admissions f
    JOIN dim_patient p ON f.patient_key = p.patient_key
    GROUP BY p.age_group
    ORDER BY p.age_group;
"""
print(pd.read_sql(q1, engine).to_string(index=False))

print("\n  KPI 2 — Average Hospital Stay by Readmission Status:")
q2 = """
    SELECT
        readmitted,
        ROUND(AVG(time_in_hospital), 2)    AS avg_days,
        ROUND(AVG(num_medications), 2)     AS avg_medications,
        ROUND(AVG(num_lab_procedures), 2)  AS avg_lab_procedures,
        COUNT(*)                           AS total_patients
    FROM fact_admissions
    GROUP BY readmitted;
"""
print(pd.read_sql(q2, engine).to_string(index=False))

print("\n  KPI 3 — Top 5 Primary Diagnoses in Readmitted Patients (<30d):")
q3 = """
    SELECT
        d.diag_1,
        COUNT(*)  AS readmitted_count
    FROM fact_admissions f
    JOIN dim_diagnosis d ON f.diagnosis_key = d.diagnosis_key
    WHERE f.readmitted = '<30'
    GROUP BY d.diag_1
    ORDER BY readmitted_count DESC
    LIMIT 5;
"""
print(pd.read_sql(q3, engine).to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  CLOSE CONNECTION
# ─────────────────────────────────────────────────────────────
cursor.close()
conn.close()

print("\n" + "="*55)
print("   ETL COMPLETE ✔")
print("="*55)
print("""
  Tables loaded in healthops_iq database:
    ✔  dim_patient       (unique patients)
    ✔  dim_diagnosis     (ICD codes + medications)
    ✔  dim_admission     (admission/discharge types)
    ✔  dim_time          (1999-2008 monthly buckets)
    ✔  fact_admissions   (one row per encounter)

  NEXT STEP → Week 2: Databricks Bronze Layer (PySpark)
""")