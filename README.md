@'
# Bhopal Air Quality Risk Predictor

A machine learning pipeline that classifies hourly air quality risk (Low / Moderate / High) for Bhopal, India, using real CPCB/MPPCB monitoring data — plus an interactive Power BI dashboard built on the model's output.

## Project Motivation

Air Quality Index (AQI) numbers are useful to scientists but not to the public. This project reframes raw pollutant readings into a **risk classification** — a direct answer to the question a resident, school, or local health authority actually needs: *"Is right now dangerous?"*

## Data Source & Scope

Data comes from the [Air Quality Data in India (2015-2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) Kaggle dataset, originally sourced from India's Central Pollution Control Board (CPCB) monitoring network.

**Important scoping note:** Madhya Pradesh has 16 registered CPCB/MPPCB monitoring stations. Before modeling, I checked data completeness across all 16 and found that **only one station -- MP001 (T.T. Nagar, Bhopal) -- has actual pollutant readings** in either the daily or hourly data files; the other 15 are registered in the system's metadata but have no recorded readings in this dataset. Rather than overclaim state-wide coverage, this project is scoped honestly as a **single-station (Bhopal) hourly risk predictor**, using ~6,900 hourly readings from Sept 2019 to Jul 2020.

## Approach

1. **Data cleaning:** Forward/back-filled short sensor gaps (<=6 hours) using per-hour autocorrelation; longer gaps were left as missing rather than guessed at, and any row without a usable AQI/AQI_Bucket was dropped.
2. **Risk labeling:** Collapsed CPCB's 6 official AQI buckets (Good to Severe) into 3 risk tiers -- **Low, Moderate, High** -- matching the decision a health/policy audience actually needs, rather than a raw AQI regression.
3. **Feature engineering:** Lag features (previous hour's pollutant levels) and 6-hour rolling averages for PM2.5, PM10, NO2, CO, SO2, O3, plus hour-of-day, month, and day-of-week.
4. **Train/test split:** Initially split chronologically (last 20% of the timeline as test data) -- but this happened to place the entire test window in a low-pollution season (May-Jul), producing **zero High-risk hours in evaluation**. I corrected this by deliberately testing on **winter months** (Dec-Feb), where Bhopal's pollution is demonstrably worst (see EDA below) -- the actual scenario the model needs to work for.
5. **Model:** Random Forest classifier (`class_weight="balanced"` to counter the rarity of High-risk hours).

## Key EDA Findings

- **Strong seasonality:** AQI spikes sharply from November through February, consistent with known winter inversion effects in North/Central India.
- **A modest daily rhythm:** Average AQI varies by only ~10 points across the 24-hour cycle (dip around 7-8 AM, peak in evening hours) -- meaningfully smaller than the seasonal swing, meaning **season, not time of day, is the dominant driver** of Bhopal's air quality.
- Full risk distribution (all data): 60.1% Moderate, 26.4% Low, 13.5% High.

## Results

Evaluated on winter hours (Dec-Feb), the season where High-risk hours actually occur:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| High | 0.830 | 0.492 | 0.618 | 455 |
| Low | 0.518 | 0.220 | 0.309 | 132 |
| Moderate | 0.821 | 0.956 | 0.883 | 1,597 |

**Overall accuracy: 81.5%**

**Top predictive features:** PM2.5 (rolling 6-hr average and previous hour), PM10 (same), and Month -- confirming the model is learning real, known air-quality signal rather than noise.

## Honest Limitations

- **High-risk recall is 49.2%** -- the model currently misses roughly half of genuinely hazardous hours. For a risk-alert use case, this is the clearest priority for improvement, not a number to round up.
- **Low-risk recall is weak (22.0%)** in the winter test set, likely because clean-air hours are rare during winter (132 of 2,184 test hours) and sit close to the Moderate boundary.
- Single-station scope (Bhopal only) -- see Data Source note above.
- Data covers Sept 2019-Jul 2020; this is historical, not live, risk modeling.

## Future Work

- Cost-sensitive/threshold tuning to trade some false alarms for better High-risk recall (a safety-relevant classifier should probably err this direction).
- Incorporate meteorological features (wind speed/direction, temperature, humidity) -- not present in this dataset but known to strongly affect pollutant dispersion.
- Extend to live CPCB API data for real-time scoring, if/when other MP stations begin reporting consistently.

## Dashboard

An interactive Power BI dashboard (`Bhopal_Air_Quality_Dashboard.pbix`) visualizes:
- Hourly AQI trend over time
- Average AQI by hour of day
- Risk class distribution
- Model predictions vs. actual risk labels (winter test set)

## Tech Stack

Python (pandas, scikit-learn, matplotlib, seaborn) - Power BI Desktop

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python bhopal_air_quality_risk_predictor.py
```

Update `DATA_DIR` in the script to point to your local copy of `station_hour.csv` and `stations.csv` from the Kaggle dataset above. The script prints EDA/evaluation output to console and saves plots plus `bhopal_air_quality_for_powerbi.csv` for the dashboard.

## Repository Structure