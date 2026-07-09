"""
Bhopal Air Quality Risk Predictor  (T.T. Nagar station, MP001)
================================================================
Scope note: Of Madhya Pradesh's 16 registered CPCB/MPPCB stations, only
MP001 (T.T. Nagar, Bhopal) has actual pollutant readings in this dataset
(verified against both station_day.csv and station_hour.csv). This
pipeline uses the HOURLY file for that one station, which gives far more
data (~6,900 hourly readings vs. 289 daily rows) and enough hazardous
hours to properly evaluate the "High risk" class -- something the daily
file could not support.

Data source: "Air Quality Data in India (2015-2020)" -> station_hour.csv
Update DATA_DIR below to wherever you've saved the Kaggle files.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = "./data/raw"   # <-- change this to your local path
STATION_ID = "MP001"      # T.T. Nagar, Bhopal -- the only MP station with real readings

# ---------------------------------------------------------------------------
# 1. LOAD & FILTER TO BHOPAL (MP001), HOURLY RESOLUTION
# ---------------------------------------------------------------------------
station_hour = pd.read_csv(f"{DATA_DIR}/station_hour.csv", parse_dates=["Datetime"])
df = station_hour[station_hour["StationId"] == STATION_ID].copy()
df = df.sort_values("Datetime").reset_index(drop=True)
print(f"Loaded {len(df)} hourly rows for {STATION_ID} (T.T. Nagar, Bhopal)")
print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")

# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING (documented, not just dropna)
# ---------------------------------------------------------------------------
pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                   "O3", "Benzene", "Toluene", "Xylene"]
pollutant_cols = [c for c in pollutant_cols if c in df.columns]

# Hourly sensor gaps are common (maintenance, calibration). Forward/back-fill
# short gaps (<=6 hours) since pollutant levels are strongly autocorrelated
# hour-to-hour; longer gaps are left as NaN and handled by the AQI dropna below
# rather than guessed at.
for col in pollutant_cols:
    df[col] = df[col].ffill(limit=6).bfill(limit=6)

before = len(df)
df = df.dropna(subset=["AQI", "AQI_Bucket"])
print(f"Dropped {before - len(df)} rows with no usable AQI/AQI_Bucket after fill.")

# ---------------------------------------------------------------------------
# 3. TARGET: RISK LABEL
# ---------------------------------------------------------------------------
# CPCB's 6 AQI buckets collapsed into a 3-level risk label -- the framing a
# health/policy audience actually needs ("is this hour dangerous?"), not a
# raw AQI regression.
risk_map = {
    "Good": "Low", "Satisfactory": "Low",
    "Moderate": "Moderate",
    "Poor": "High", "Very Poor": "High", "Severe": "High",
}
df["Risk"] = df["AQI_Bucket"].map(risk_map)
print("\nRisk class balance:\n", df["Risk"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["Hour"] = df["Datetime"].dt.hour
df["Month"] = df["Datetime"].dt.month
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Season"] = df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
})

# Lag features: the previous hour's readings, plus a short rolling window,
# are strong predictors of the current hour's risk.
lag_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
lag_cols = [c for c in lag_cols if c in df.columns]
for col in lag_cols:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_roll6"] = df[col].shift(1).rolling(6, min_periods=1).mean()

feature_lag_cols = [f"{c}_lag1" for c in lag_cols] + [f"{c}_roll6" for c in lag_cols]
df = df.dropna(subset=feature_lag_cols)

# ---------------------------------------------------------------------------
# 5. EDA
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
monthly_avg = df.groupby("Month")["AQI"].mean().reset_index()
sns.barplot(data=monthly_avg, x="Month", y="AQI", color="steelblue")
plt.title("Bhopal (T.T. Nagar) — Average AQI by Month")
plt.tight_layout()
plt.savefig("bhopal_aqi_seasonality.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
hourly_avg = df.groupby("Hour")["AQI"].mean().reset_index()
sns.lineplot(data=hourly_avg, x="Hour", y="AQI", marker="o")
plt.title("Bhopal (T.T. Nagar) — Average AQI by Hour of Day")
plt.tight_layout()
plt.savefig("bhopal_aqi_by_hour.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Risk", order=["Low", "Moderate", "High"], palette="Blues_d")
plt.title("Bhopal — Risk Class Counts (full dataset)")
plt.tight_layout()
plt.savefig("bhopal_risk_counts.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT -- season-based, not last-N%-of-timeline
# ---------------------------------------------------------------------------
test_mask = df["Season"] == "Winter"
train = df[~test_mask]
test = df[test_mask]
print(f"\nTrain (non-winter): {len(train)} rows")
print(f"Test  (winter):      {len(test)} rows")
print("\nTest-set risk class counts (confirms whether High is represented):")
print(test["Risk"].value_counts())

feature_cols = feature_lag_cols + ["Hour", "Month", "DayOfWeek"]
X_train, y_train = train[feature_cols], train["Risk"]
X_test, y_test = test[feature_cols], test["Risk"]

# ---------------------------------------------------------------------------
# 7. MODEL
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",  # High-risk hours are rarer -- don't let the model ignore them
    random_state=42,
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# ---------------------------------------------------------------------------
# 8. EVALUATION -- recall on "High" matters more than overall accuracy
# ---------------------------------------------------------------------------
print("\nClassification report:\n")
print(classification_report(y_test, preds, digits=3))

labels_present = [l for l in ["Low", "Moderate", "High"] if l in set(y_test) | set(preds)]
cm = confusion_matrix(y_test, preds, labels=labels_present)
disp = ConfusionMatrixDisplay(cm, display_labels=labels_present)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Bhopal Hourly Risk Classifier")
plt.tight_layout()
plt.savefig("bhopal_confusion_matrix.png", dpi=150)
plt.close()

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:\n", importances.head(10))

# ---------------------------------------------------------------------------
# 9. EXPORT FOR POWER BI
# ---------------------------------------------------------------------------
export_cols = ["Datetime", "StationId", "AQI", "AQI_Bucket", "Risk", "Hour", "Month"] + pollutant_cols
export_df = df[[c for c in export_cols if c in df.columns]].copy()
export_df["Predicted_Risk"] = None
export_df.loc[test.index, "Predicted_Risk"] = preds
export_df.to_csv("bhopal_air_quality_for_powerbi.csv", index=False)
print("\nSaved bhopal_air_quality_for_powerbi.csv for your Power BI dashboard.")
