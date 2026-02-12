import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np
import re

# -----------------------------
# 1. Load log file
# -----------------------------
log_file = "g_lomidze25_63947_server.log"

timestamps = []

# Robust timestamp parsing
with open(log_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        ts = None
        # Try common web log format: [12/Feb/2026:12:01:45 +0000]
        m = re.search(r'\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2})', line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%d/%b/%Y:%H:%M:%S")
            except:
                ts = None
        else:
            # Try ISO format: 2026-02-12 12:01:45
            m2 = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if m2:
                try:
                    ts = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S")
                except:
                    ts = None
        if ts:
            timestamps.append(ts)

print(f"Parsed {len(timestamps)} valid timestamps")

if len(timestamps) == 0:
    print("No valid timestamps found. Check log file format.")
    exit()

# -----------------------------
# 2. Aggregate requests per second
# -----------------------------
df = pd.DataFrame({'timestamp': timestamps})
df['count'] = 1

# Aggregate per second
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).reset_index(drop=True)
df = df.groupby('timestamp').count().reset_index()

# Seconds since first timestamp for regression
df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

# -----------------------------
# 3. Linear regression
# -----------------------------
X = df['seconds'].values.reshape(-1, 1)
y = df['count'].values

reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

# -----------------------------
# 4. Detect DDoS intervals
# -----------------------------
threshold = y_pred.mean() + 3 * y_pred.std()
ddos_times = df[df['count'] > threshold]['timestamp']

intervals = []
if not ddos_times.empty:
    start = ddos_times.iloc[0]
    prev = start
    for t in ddos_times[1:]:
        if (t - prev).total_seconds() > 1:
            intervals.append((start, prev))
            start = t
        prev = t
    intervals.append((start, prev))

print("Detected DDoS intervals:")
for s, e in intervals:
    print(s, "→", e)

# -----------------------------
# 5. Visualization
# -----------------------------
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['timestamp'], y=df['count'], label="Requests per second")
sns.lineplot(x=df['timestamp'], y=y_pred, label="Regression line", color="red")

# Highlight DDoS intervals
for s, e in intervals:
    plt.axvspan(s, e, color='orange', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Number of Requests")
plt.title("Web Server Request Rate with Regression Line and DDoS Intervals")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("requests_regression.png")
plt.show()
