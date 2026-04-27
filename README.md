# 📊 Time Series Forecasting & Anomaly Detection

## 🚀 Overview

This project builds an end-to-end machine learning pipeline to:

* Forecast Wikipedia page traffic using time series data
* Detect anomalies using forecast residuals
* Compare baseline and feature-engineered models

The project demonstrates how forecasting models can be combined with statistical techniques to identify abnormal patterns in real-world data.

---

## 📂 Dataset

* Source: Wikipedia Web Traffic Dataset (Kaggle)
* Contains daily page views for Wikipedia articles

⚠️ Dataset not included due to size. Download from Kaggle and place `train_1.csv` in the project root directory.

---

## 🧠 Approach

### 1. Data Preprocessing

* Converted data from wide format to time series format
* Parsed date columns into datetime format
* Handled missing values and sorted chronologically

---

### 2. Exploratory Data Analysis (EDA)

* Visualized time series trends and sudden spikes
* Analyzed distribution of page views (high variance observed)
* Identified weekly patterns using day-of-week aggregation
* Used autocorrelation to understand temporal dependencies

---

### 3. Feature Engineering

#### Baseline Features

* Lag features: `lag_1`, `lag_7`, `lag_14`
* Rolling statistics: mean and standard deviation (7-day window)
* Date features: day of week, month

#### Improved Features

* Additional lags: `lag_2`, `lag_3`, `lag_30`
* Multiple rolling windows (3, 7, 14 days)
* Rolling maximum feature
* Difference feature (`diff_1`) to capture sudden changes

---

### 4. Modeling

Used XGBoost regression for time series forecasting.

* Baseline model trained on basic features
* Improved model trained on extended feature set

---

### 5. Anomaly Detection

Anomalies are detected using forecast residuals:

```
residual = actual - predicted
```

Residuals are standardized using Z-score:

```
|z-score| > 3 → anomaly
```

---

## 📈 Results

* Baseline Model MAE: ~11.18
* Improved Model MAE: ~2.16

Key observations:

* Improved model significantly reduced prediction error
* Large traffic spikes were successfully detected as anomalies
* Some false positives occurred after extreme spikes due to model lag

---

## ⚠️ Key Insights

* Time series data is highly noisy with irregular spikes
* Feature engineering plays a critical role in improving model performance
* Improved forecasting reduces residual magnitude, which can weaken anomaly signals
* Residual-based anomaly detection may produce false positives after extreme events

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* XGBoost


