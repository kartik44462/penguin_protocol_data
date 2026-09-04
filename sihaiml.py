import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mysql.connector


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# THIS IS THE ONLY INPUT DATASET
INPUT_FILE = os.path.join(
    BASE_DIR,
    "Hourly_antarctic_telemetry.csv"
)

# THIS IS ONLY AN OUTPUT FILE
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Antarctic Digital Twin Prediction.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ANTARCTICA DIGITAL TWIN - DATA LOADING")
print("=" * 60)

try:
    df = pd.read_csv(INPUT_FILE)
    print("\nDataset loaded successfully!")
    print(f"Input file: {INPUT_FILE}")
except FileNotFoundError:
    print("\nERROR: Hourly_antarctic_telemetry.csv was not found.")
    print("Make sure it is in the same folder as sihaiml.py")
    exit()


print("\nFirst 5 rows:")
print(df.head())

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove completely empty rows
df = df.dropna(how="all")

# Remove rows where essential ML values are missing
required_columns = [
    "temperature_celsius",
    "wind_speed_knots",
    "generator_load_percent",
    "energy_consumed_kwh",
    "battery_level_percent",
    "fuel_level_liters",
    "station_occupancy"
]

df = df.dropna(subset=required_columns)

# Convert numerical columns to numeric
numeric_columns = [
    "temperature_celsius",
    "wind_speed_knots",
    "pressure_hpa",
    "humidity_percent",
    "generator_load_percent",
    "energy_consumed_kwh",
    "battery_level_percent",
    "fuel_level_liters",
    "fuel_burn_rate_lph",
    "food_inventory_kg",
    "station_occupancy"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows that became invalid
df = df.dropna(subset=required_columns)

print("Cleaned dataset successfully!")
print(f"Number of records: {len(df)}")


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("PREPARING ML FEATURES")
print("=" * 60)

"""
Your actual CSV does NOT contain Solar_Radiation.

Therefore, we do not pretend that the source dataset contains it.
For compatibility with the existing model structure, we create
a Solar_Radiation feature with value 0.

Later, if you obtain actual solar radiation data, replace this.
"""

model_df = pd.DataFrame({
    "Temperature": df["temperature_celsius"],
    "Wind_Speed": df["wind_speed_knots"],
    "Solar_Radiation": 0,
    "Occupancy": df["station_occupancy"],
    "Battery_Level": df["battery_level_percent"],
    "Generator_Load": df["generator_load_percent"],
    "Fuel_Level": df["fuel_level_liters"]
})

target = df["energy_consumed_kwh"]

features = [
    "Temperature",
    "Wind_Speed",
    "Solar_Radiation",
    "Occupancy",
    "Battery_Level",
    "Generator_Load",
    "Fuel_Level"
]

print("\nFeatures used:")
print(features)

print("\nTarget:")
print("Energy_Consumption")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    model_df,
    target,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 6. RANDOM FOREST ENERGY MODEL
# ============================================================

print("\n" + "=" * 60)
print("ENERGY PREDICTION MODEL")
print("=" * 60)

energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

energy_model.fit(X_train, y_train)

y_pred = energy_model.predict(X_test)


# ============================================================
# 7. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEnergy Model Performance")
print("-" * 40)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": energy_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.to_string(index=False))


# ============================================================
# 9. PREDICT ENERGY FOR ALL DATA
# ============================================================

df["predicted_energy"] = energy_model.predict(model_df)


# ============================================================
# 10. ISOLATION FOREST ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY DETECTION")
print("=" * 60)

anomaly_model = IsolationForest(
    contamination=0.10,
    random_state=42
)

anomaly_model.fit(model_df)

df["anomaly_prediction"] = anomaly_model.predict(model_df)

df["anomaly_status"] = np.where(
    df["anomaly_prediction"] == -1,
    "ANOMALY",
    "NORMAL"
)

print(
    f"Anomalies detected: "
    f"{(df['anomaly_status'] == 'ANOMALY').sum()}"
)


# ============================================================
# 11. CURRENT STATION CONDITIONS
# ============================================================

print("\n" + "=" * 60)
print("CURRENT STATION PREDICTION")
print("=" * 60)

current_station = pd.DataFrame({
    "Temperature": [-25],
    "Wind_Speed": [30],
    "Solar_Radiation": [150],
    "Occupancy": [25],
    "Battery_Level": [65],
    "Generator_Load": [60],
    "Fuel_Level": [500]
})

current_energy_prediction = energy_model.predict(
    current_station
)[0]

print(f"\nPredicted Energy Consumption: "
      f"{current_energy_prediction:.2f} kWh")


# ============================================================
# 12. ENERGY STATUS
# ============================================================

battery_capacity = 1000

current_battery = 65 / 100 * battery_capacity

if current_energy_prediction > current_battery:
    energy_status = "CRITICAL"
elif current_energy_prediction > current_battery * 0.7:
    energy_status = "WARNING"
else:
    energy_status = "NORMAL"

print("\nEnergy Status:", energy_status)


# ============================================================
# 13. ENERGY ENDURANCE
# ============================================================

if current_energy_prediction > 0:
    energy_endurance = current_battery / current_energy_prediction
else:
    energy_endurance = 0

print(
    f"Energy Endurance: "
    f"{energy_endurance:.2f} days"
)


# ============================================================
# 14. FUEL STATUS
# ============================================================

current_fuel = 500
fuel_consumption_per_day = 70

fuel_endurance = (
    current_fuel / fuel_consumption_per_day
)

if fuel_endurance < 3:
    fuel_status = "CRITICAL"
elif fuel_endurance < 7:
    fuel_status = "WARNING"
else:
    fuel_status = "NORMAL"

print("\nFuel Status:", fuel_status)

print(
    f"Fuel Remaining: "
    f"{current_fuel:.2f} litres"
)

print(
    f"Fuel Endurance: "
    f"{fuel_endurance:.2f} days"
)


# ============================================================
# 15. FUEL ALERT
# ============================================================

if fuel_endurance < 7:
    fuel_alert = "FUEL SUPPLY REQUIRED"
else:
    fuel_alert = "FUEL LEVEL SUFFICIENT"

print("\nFuel Alert:", fuel_alert)


# ============================================================
# 16. ENERGY ALERT
# ============================================================

if energy_endurance < 5:
    energy_alert = "ENERGY RESERVE LOW"
else:
    energy_alert = "ENERGY LEVEL SUFFICIENT"

print("Energy Alert:", energy_alert)


# ============================================================
# 17. CURRENT ANOMALY
# ============================================================

current_anomaly_result = anomaly_model.predict(
    current_station
)[0]

if current_anomaly_result == -1:
    current_anomaly_status = "ANOMALY"
else:
    current_anomaly_status = "NORMAL"

print("\nCurrent Anomaly Status:", current_anomaly_status)


# ============================================================
# 18. STATION RISK SCORE
# ============================================================

risk_score = 0

if fuel_endurance < 7:
    risk_score += 30

if energy_endurance < 5:
    risk_score += 30

if current_anomaly_status == "ANOMALY":
    risk_score += 40

risk_score = min(risk_score, 100)

if risk_score >= 70:
    risk_status = "HIGH RISK"
elif risk_score >= 40:
    risk_status = "MEDIUM RISK"
else:
    risk_status = "LOW RISK"

print("\n" + "=" * 60)
print("STATION RISK")
print("=" * 60)

print(f"Risk Score: {risk_score} / 100")
print(f"Risk Status: {risk_status}")


# ============================================================
# 19. NEXT ANTARCTICA VISIT PLAN
# ============================================================

planning_days = 30
reserve = 0.20

required_fuel = (
    planning_days *
    fuel_consumption_per_day *
    (1 + reserve)
)

additional_fuel = max(
    0,
    required_fuel - current_fuel
)

print("\n" + "=" * 60)
print("NEXT ANTARCTICA VISIT PLAN")
print("=" * 60)

print(
    f"Planning Period: "
    f"{planning_days} days"
)

print(
    f"Recommended Fuel for Next Visit: "
    f"{required_fuel:.2f} litres"
)

print(
    f"Additional Fuel Required: "
    f"{additional_fuel:.2f} litres"
)


# ============================================================
# 20. ENERGY RECOMMENDATION
# ============================================================

if energy_status == "CRITICAL":
    energy_recommendation = (
        "Immediately reduce non-essential energy consumption."
    )
elif energy_status == "WARNING":
    energy_recommendation = (
        "Monitor battery usage and reduce unnecessary loads."
    )
else:
    energy_recommendation = (
        "Energy consumption is within acceptable limits."
    )

print("\nEnergy Recommendation:")
print(energy_recommendation)


# ============================================================
# 21. MAINTENANCE RECOMMENDATION
# ============================================================

if current_anomaly_status == "ANOMALY":
    maintenance_recommendation = (
        "Immediate equipment inspection recommended."
    )
elif risk_score >= 40:
    maintenance_recommendation = (
        "Schedule preventive maintenance before next visit."
    )
else:
    maintenance_recommendation = (
        "Continue routine preventive maintenance."
    )

print("\nMaintenance Recommendation:")
print(maintenance_recommendation)


# ============================================================
# 22. ADD CALCULATED VALUES TO DATASET
# ============================================================

df["energy_remaining_kwh"] = current_battery
df["energy_endurance_days"] = energy_endurance
df["fuel_endurance_days"] = fuel_endurance
df["risk_score"] = risk_score
df["recommended_fuel_litres"] = required_fuel
df["energy_status"] = energy_status
df["fuel_status"] = fuel_status


# ============================================================
# 23. FINAL DIGITAL TWIN REPORT
# ============================================================

print("\n")
print("=" * 60)
print("       ANTARCTICA DIGITAL TWIN REPORT")
print("=" * 60)

print(f"Station: MAITRI")

print(
    f"Predicted Energy Consumption: "
    f"{current_energy_prediction:.2f} kWh"
)

print(
    f"Energy Endurance: "
    f"{energy_endurance:.2f} days"
)

print(
    f"Fuel Remaining: "
    f"{current_fuel:.2f} litres"
)

print(
    f"Fuel Endurance: "
    f"{fuel_endurance:.2f} days"
)

print(
    f"Anomaly Status: "
    f"{current_anomaly_status}"
)

print(
    f"Risk Score: "
    f"{risk_score} / 100"
)

print(
    f"Recommended Fuel for Next Visit: "
    f"{required_fuel:.2f} litres"
)

print("=" * 60)


# ============================================================
# 24. SAVE PREDICTIONS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPrediction file saved successfully!")
print(f"Output: {OUTPUT_FILE}")


# ============================================================
# 25. MYSQL CONNECTION
# ============================================================

print("\n" + "=" * 60)
print("MYSQL DATABASE")
print("=" * 60)

try:

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="lavanya",
        database="antarctica"
    )

    cursor = connection.cursor()

    print("Connected to MySQL!")

    insert_query = """
    INSERT INTO predictions (
        timestamp,
        temperature,
        wind_speed,
        solar_radiation,
        occupancy,
        battery_level,
        generator_load,
        fuel_level,
        predicted_energy,
        energy_remaining,
        energy_endurance,
        fuel_endurance,
        anomaly_status,
        risk_score,
        recommended_fuel,
        energy_status,
        fuel_status
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    current_timestamp = pd.Timestamp.now()

    values = (
        current_timestamp,
        -25,
        30,
        150,
        25,
        65,
        60,
        500,
        float(current_energy_prediction),
        float(current_battery),
        float(energy_endurance),
        float(fuel_endurance),
        current_anomaly_status,
        int(risk_score),
        float(required_fuel),
        energy_status,
        fuel_status
    )

    cursor.execute(insert_query, values)

    connection.commit()

    print("Prediction successfully saved to MySQL!")

    cursor.close()
    connection.close()

except mysql.connector.Error as error:

    print("\nMySQL Error:")
    print(error)


# ============================================================
# 26. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("DIGITAL TWIN PROCESS COMPLETED")
print("=" * 60)