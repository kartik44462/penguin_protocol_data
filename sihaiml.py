# ============================================================
# ANTARCTICA DIGITAL TWIN - RESOURCE & RISK MODEL
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("/Users/lavanya/antarctica_data.csv")

print("Dataset loaded successfully!")
print(data.head())


# ============================================================
# 2. ENERGY CONSUMPTION MODEL
# ============================================================

features = [
    "Temperature",
    "Wind_Speed",
    "Solar_Radiation",
    "Occupancy",
    "Battery_Level",
    "Generator_Load",
    "Fuel_Level"
]

X = data[features]
y = data["Energy_Consumption"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
energy_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train
energy_model.fit(X_train, y_train)


# Predict
y_pred = energy_model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("ENERGY MODEL PERFORMANCE")
print("==============================")

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 2))


# ============================================================
# 3. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": energy_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance)


# ============================================================
# 4. CURRENT ANTARCTICA STATION CONDITIONS
# ============================================================

current_station = pd.DataFrame({
    "Temperature": [-25],
    "Wind_Speed": [30],
    "Solar_Radiation": [150],
    "Occupancy": [25],
    "Battery_Level": [65],
    "Generator_Load": [60],
    "Fuel_Level": [500]
})


# ============================================================
# 5. PREDICT ENERGY CONSUMPTION
# ============================================================

predicted_energy = energy_model.predict(
    current_station
)[0]

print("\n==============================")
print("CURRENT STATION PREDICTION")
print("==============================")

print(
    "Predicted Energy Consumption:",
    round(predicted_energy, 2)
)


# ============================================================
# 6. REMAINING ENERGY
# ============================================================

# Example battery capacity
battery_capacity = 1000       # kWh

battery_percentage = current_station[
    "Battery_Level"
].iloc[0]

energy_available = (
    battery_percentage / 100
) * battery_capacity


# Assume solar generation for the current period
solar_generation = 150         # kWh


total_energy_available = (
    energy_available + solar_generation
)

energy_left = (
    total_energy_available - predicted_energy
)

print("\n==============================")
print("ENERGY STATUS")
print("==============================")

print(
    "Battery Energy Available:",
    round(energy_available, 2),
    "kWh"
)

print(
    "Solar Energy Generated:",
    solar_generation,
    "kWh"
)

print(
    "Predicted Energy Consumption:",
    round(predicted_energy, 2),
    "kWh"
)

print(
    "Estimated Energy Left:",
    round(energy_left, 2),
    "kWh"
)


# ============================================================
# 7. ESTIMATE ENERGY ENDURANCE
# ============================================================

if predicted_energy > 0:

    days_of_energy = (
        total_energy_available /
        predicted_energy
    )

else:

    days_of_energy = float("inf")


print(
    "Estimated Energy Endurance:",
    round(days_of_energy, 2),
    "days"
)


# ============================================================
# 8. REMAINING FUEL
# ============================================================

current_fuel = current_station[
    "Fuel_Level"
].iloc[0]

# Example generator fuel consumption rate
generator_fuel_rate = 70      # litres/day


fuel_days_remaining = (
    current_fuel /
    generator_fuel_rate
)


print("\n==============================")
print("FUEL STATUS")
print("==============================")

print(
    "Remaining Fuel:",
    current_fuel,
    "litres"
)

print(
    "Daily Fuel Consumption:",
    generator_fuel_rate,
    "litres/day"
)

print(
    "Estimated Fuel Endurance:",
    round(fuel_days_remaining, 2),
    "days"
)


# ============================================================
# 9. FUEL ALERT
# ============================================================

if fuel_days_remaining < 3:

    fuel_status = "CRITICAL"

elif fuel_days_remaining < 7:

    fuel_status = "WARNING"

else:

    fuel_status = "NORMAL"


print(
    "Fuel Status:",
    fuel_status
)


# ============================================================
# 10. ENERGY ALERT
# ============================================================

if days_of_energy < 2:

    energy_status = "CRITICAL"

elif days_of_energy < 5:

    energy_status = "WARNING"

else:

    energy_status = "NORMAL"


print(
    "Energy Status:",
    energy_status
)


# ============================================================
# 11. ANOMALY DETECTION
# ============================================================

anomaly_features = [
    "Temperature",
    "Wind_Speed",
    "Solar_Radiation",
    "Occupancy",
    "Battery_Level",
    "Generator_Load",
    "Fuel_Level",
    "Energy_Consumption"
]

anomaly_model = IsolationForest(
    contamination=0.10,
    random_state=42
)

anomaly_model.fit(
    data[anomaly_features]
)


# Check current station
current_energy = predicted_energy

current_anomaly_data = current_station.copy()

current_anomaly_data[
    "Energy_Consumption"
] = current_energy


anomaly_result = anomaly_model.predict(
    current_anomaly_data[anomaly_features]
)[0]


print("\n==============================")
print("ANOMALY DETECTION")
print("==============================")


if anomaly_result == -1:

    print("⚠️ ANOMALY DETECTED")
    anomaly_status = "ANOMALY"

else:

    print("✅ Station operating normally")
    anomaly_status = "NORMAL"


# ============================================================
# 12. SIMPLE RISK SCORE
# ============================================================

risk_score = 0


if fuel_days_remaining < 7:
    risk_score += 30

if days_of_energy < 5:
    risk_score += 30

if anomaly_status == "ANOMALY":
    risk_score += 40


print("\n==============================")
print("STATION RISK")
print("==============================")

print(
    "Risk Score:",
    risk_score,
    "/ 100"
)


if risk_score >= 70:

    print("🔴 HIGH RISK")

elif risk_score >= 40:

    print("🟠 MEDIUM RISK")

else:

    print("🟢 LOW RISK")


# ============================================================
# 13. NEXT VISIT RESOURCE RECOMMENDATION
# ============================================================

print("\n==============================")
print("NEXT ANTARCTICA VISIT PLAN")
print("==============================")


# Planning horizon
planning_days = 30


# Fuel required for planning period
fuel_required = (
    generator_fuel_rate *
    planning_days
)

# Safety reserve
fuel_reserve = (
    fuel_required * 0.20
)

total_fuel_required = (
    fuel_required +
    fuel_reserve
)

fuel_to_bring = max(
    0,
    total_fuel_required - current_fuel
)


print(
    "Recommended Fuel:",
    round(fuel_to_bring, 2),
    "litres"
)


# ============================================================
# 14. ENERGY RESOURCE RECOMMENDATION
# ============================================================

if energy_status == "CRITICAL":

    print("🔋 Priority: Additional battery capacity")

elif energy_status == "WARNING":

    print("🔋 Recommendation: Inspect battery and solar system")

else:

    print("🔋 Energy resources: No immediate action")


# ============================================================
# 15. MAINTENANCE RECOMMENDATION
# ============================================================

if anomaly_status == "ANOMALY":

    print("🔧 Priority: Generator/equipment inspection")

else:

    print("🔧 Maintenance: Routine inspection")


# ============================================================
# 16. FINAL DIGITAL TWIN REPORT
# ============================================================

print("\n")
print("==========================================")
print("      ANTARCTICA DIGITAL TWIN REPORT")
print("==========================================")

print(
    "Predicted Energy:",
    round(predicted_energy, 2),
    "kWh"
)

print(
    "Energy Remaining:",
    round(energy_left, 2),
    "kWh"
)

print(
    "Energy Endurance:",
    round(days_of_energy, 2),
    "days"
)

print(
    "Fuel Remaining:",
    current_fuel,
    "litres"
)

print(
    "Fuel Endurance:",
    round(fuel_days_remaining, 2),
    "days"
)

print(
    "Anomaly Status:",
    anomaly_status
)

print(
    "Risk Score:",
    risk_score,
    "/ 100"
)

print(
    "Recommended Fuel for Next Visit:",
    round(fuel_to_bring, 2),
    "litres"
)

print("==========================================")
# ============================================================
# 17. SAVE DIGITAL TWIN OUTPUT TO MYSQL
# ============================================================

import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="lavanya",
    database="antarctica"
)

cursor = db.cursor()

print("Connected to MySQL!")

sql = """
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
    NOW(), %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s
)
"""

values = (
    float(current_station["Temperature"].iloc[0]),
    float(current_station["Wind_Speed"].iloc[0]),
    float(current_station["Solar_Radiation"].iloc[0]),
    int(current_station["Occupancy"].iloc[0]),
    float(current_station["Battery_Level"].iloc[0]),
    float(current_station["Generator_Load"].iloc[0]),
    float(current_station["Fuel_Level"].iloc[0]),

    float(predicted_energy),
    float(energy_left),
    float(days_of_energy),
    float(fuel_days_remaining),

    anomaly_status,
    int(risk_score),
    float(fuel_to_bring),

    energy_status,
    fuel_status
)

cursor.execute(sql, values)
db.commit()

print("✅ Digital Twin prediction saved to MySQL!")

cursor.close()
db.close()