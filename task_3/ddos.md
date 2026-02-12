# DDoS Attack Detection Using Regression Analysis

## 1. Event Log File

The analysis uses the web server log file:

[g_lomidze25_63947_server.log](g_lomidze25_63947_server.log)

---

## 2. Objective

The goal of this task is to **detect potential DDoS attack intervals** by analyzing web server request logs. Regression analysis models normal traffic, and deviations above a threshold indicate abnormal activity.

---

## 3. Methodology

1. **Load the log file** and parse timestamps in multiple common formats:  
   - `[12/Feb/2026:12:01:45]`  
   - `2026-02-12 12:01:45`  

2. **Aggregate requests per second** to create a time series of server load.

3. **Perform regression analysis** to model normal traffic trends:

```python
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
