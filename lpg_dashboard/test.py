# test_load.py
import sys
sys.path.append('C:\\Users\\priti\\Documents\\LPGSupplyAnalysis\\lpg_dashboard')

from utils.data_loader import load_all_data

data = load_all_data()

print("=== Data Loaded Successfully ===")
print(f"Shock Severity: {len(data['shock_severity'])} rows" if data['shock_severity'] is not None else "Shock Severity: None")
print(f"Combined Country Anomalies: {len(data['combined_country_anomalies'])} rows" if data['combined_country_anomalies'] is not None else "Combined Country Anomalies: None")

if data['combined_country_anomalies'] is not None:
    print("\n=== Combined Country Anomalies Columns ===")
    print(data['combined_country_anomalies'].columns.tolist())
    print("\n=== Sample ===")
    print(data['combined_country_anomalies'].head())