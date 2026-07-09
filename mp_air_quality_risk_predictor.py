"""
MP Air Quality Risk Predictor
==============================
Pipeline: filter to Madhya Pradesh stations -> clean -> feature engineer
-> train a risk classifier -> evaluate with a focus on catching hazardous days
-> export a clean file for a Power BI dashboard.

Data source: "Air Quality Data in India (2015-2020)" (station_day.csv, stations.csv)
Update DATA_DIR below to wherever you've saved the Kaggle files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "./data/raw"


# ---------------------------------------------------------------------------
# 1. LOAD & FILTER TO MP STATIONS
# ---------------------------------------------------------------------------
stations = pd.read_csv(f"{DATA_DIR}/stations.csv")
station_day = pd.read_csv(f"{DATA_DIR}/station_day.csv", parse_dates=["Date"])

mp_stations = stations[stations["State"] == "Madhya Pradesh"]
print(f"MP stations found: {len(mp_stations)}")
print(mp_stations[["StationId", "StationName", "City"]])

mp_ids = mp_stations["StationId"].tolist()
df = station_day[station_day["StationId"].isin(mp_ids)].copy()

# Attach city name so we can group/compare by city later
df = df.merge(mp_stations[["StationId", "City"]], on="StationId", how="left")

# ---------------------------------------------------------------------------
# 2. DATA AVAILABILITY CHECK  (do this before you trust any model on a station)
# ---------------------------------------------------------------------------
availability = df.groupby("StationId")["Date"].agg(
    first_date="min", last_date="max", n_records="count"
)
availability["expected_days"] = (
    (availability["last_date"] - availability["first_date"]).dt.days + 1
)
availability["completeness_pct"] = (
    100 * availability["n_records"] / availability["expected_days"]
).round(1)
print("\nData availability by station:\n", availability)

# Drop stations with unusably sparse history (tune this threshold as you see fit)
usable_ids = availability[availability["completeness_pct"] >= 40].index.tolist()
dropped = set(mp_ids) - set(usable_ids)
if dropped:
    print(f"\nDropping stations with <40% completeness: {dropped}")
df = df[df["StationId"].isin(usable_ids)]

# ---------------------------------------------------------------------------
# 3. MISSING VALUE HANDLING  (documented, not just dropna)
# ---------------------------------------------------------------------------
pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                   "O3", "Benzene", "Toluene", "Xylene"]

df = df.sort_values(["StationId", "Date"])

# Strategy: forward-fill then back-fill within each station (max 3-day gap),
# since pollutant levels are strongly autocorrelated day-to-day.
# Any remaining gaps (long stretches of missing sensor data) get dropped
# rather than guessed at, since AQI/AQI_Bucket must reflect real readings.
for col in pollutant_cols:
    if col in df.columns:
        df[col] = df.groupby("StationId")[col].transform(
            lambda s: s.ffill(limit=3).bfill(limit=3)
        )

before = len(df)
df = df.dropna(subset=["AQI", "AQI_Bucket"])
print(f"\nDropped {before - len(df)} rows with no usable AQI/AQI_Bucket after fill.")

# ---------------------------------------------------------------------------
# 4. TARGET: RISK LABEL
# ---------------------------------------------------------------------------
# AQI_Bucket already gives CPCB's official categories:
# Good, Satisfactory, Moderate, Poor, Very Poor, Severe
# We frame this as a RISK problem (not just AQI regression) because that's
# the decision a health/policy audience actually needs: "is today dangerous?"
risk_map = {
    "Good": "Low",
    "Satisfactory": "Low",
    "Moderate": "Moderate",
    "Poor": "High",
    "Very Poor": "High",
    "Severe": "High",
}
df["Risk"] = df["AQI_Bucket"].map(risk_map)
print("\nRisk class balance:\n", df["Risk"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["Month"] = df["Date"].dt.month
df["Season"] = df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
})
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Lag features: yesterday's readings are strong predictors of today's risk.
lag_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
for col in lag_cols:
    if col in df.columns:
        df[f"{col}_lag1"] = df.groupby("StationId")[col].shift(1)
        df[f"{col}_roll3"] = (
            df.groupby("StationId")[col]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )

df = df.dropna(subset=[f"{c}_lag1" for c in lag_cols if f"{c}_lag1" in df.columns])

# ---------------------------------------------------------------------------
# 6. EDA — a couple of the plots worth having in your writeup
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
monthly_avg = df.groupby(["City", "Month"])["AQI"].mean().reset_index()
sns.lineplot(data=monthly_avg, x="Month", y="AQI", hue="City")
plt.title("Average AQI by Month — MP Cities")
plt.tight_layout()
plt.savefig("mp_aqi_seasonality.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="City", hue="Risk", order=df["City"].value_counts().index)
plt.xticks(rotation=45, ha="right")
plt.title("Risk Day Counts by City")
plt.tight_layout()
plt.savefig("mp_risk_by_city.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. TRAIN / TEST SPLIT — chronological, not random
# ---------------------------------------------------------------------------
# Random shuffling would leak future information into training (e.g. a
# rolling average computed from a day that's technically "in the future"
# relative to a training row). Split by date instead.
cutoff_date = df["Date"].quantile(0.8)  # last 20% of the timeline is the test set
train = df[df["Date"] <= cutoff_date]
test = df[df["Date"] > cutoff_date]
print(f"\nTrain: {len(train)} rows through {train['Date'].max().date()}")
print(f"Test:  {len(test)} rows from {test['Date'].min().date()}")

feature_cols = (
    [f"{c}_lag1" for c in lag_cols if f"{c}_lag1" in df.columns]
    + [f"{c}_roll3" for c in lag_cols if f"{c}_roll3" in df.columns]
    + ["Month", "DayOfWeek"]
)

le_city = LabelEncoder()
df["City_enc"] = le_city.fit_transform(df["City"])
feature_cols.append("City_enc")
train["City_enc"] = le_city.transform(train["City"])
test["City_enc"] = le_city.transform(test["City"])

X_train, y_train = train[feature_cols], train["Risk"]
X_test, y_test = test[feature_cols], test["Risk"]

# ---------------------------------------------------------------------------
# 8. MODEL
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",  # High-risk days are rarer — don't let the model ignore them
    random_state=42,
)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# ---------------------------------------------------------------------------
# 9. EVALUATION — recall on "High" risk matters more than overall accuracy
# ---------------------------------------------------------------------------
print("\nClassification report:\n")
print(classification_report(y_test, preds, digits=3))

cm = confusion_matrix(y_test, preds, labels=["Low", "Moderate", "High"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "Moderate", "High"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — MP Risk Classifier")
plt.tight_layout()
plt.savefig("mp_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance — useful for the "what drives risk" section of your writeup
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:\n", importances.head(10))

# ---------------------------------------------------------------------------
# 10. EXPORT FOR POWER BI
# ---------------------------------------------------------------------------
export_cols = ["Date", "City", "StationId", "AQI", "AQI_Bucket", "Risk"] + pollutant_cols
export_df = df[[c for c in export_cols if c in df.columns]].copy()
export_df["Predicted_Risk"] = None
test_idx = test.index
export_df.loc[test_idx, "Predicted_Risk"] = preds
export_df.to_csv("mp_air_quality_for_powerbi.csv", index=False)
print("\nSaved mp_air_quality_for_powerbi.csv for your Power BI dashboard.")