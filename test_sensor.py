import pandas as pd
import numpy as np
from agents.sentinel import analyze_sensor_data
from agents.analyst import analyze_root_cause
from agents.planner import create_maintenance_plan

np.random.seed(42)
n = 100
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
    'temperature_C': np.random.normal(75, 2, n),
    'vibration_mms': np.random.normal(0.5, 0.1, n),
    'pressure_PSI': np.random.normal(100, 5, n),
})
df.loc[95:, 'temperature_C'] += 20
df.loc[95:, 'vibration_mms'] *= 4

print("=== Agent 1: SENTINEL ===")
anomaly_report = analyze_sensor_data(df)
print(anomaly_report)

print("\n=== Agent 2: ANALYST ===")
root_cause = analyze_root_cause(anomaly_report, "Centrifugal Pump")
print(root_cause)

print("\n=== Agent 3: PLANNER ===")
plan = create_maintenance_plan(root_cause, "Centrifugal Pump")
print(plan)