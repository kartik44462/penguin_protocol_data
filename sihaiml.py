# ============================================================
# ANTARCTICA DIGITAL TWIN - RESOURCE & RISK MODEL
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

# Automatically find the CSV in the same folder as this Python file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "Hourly_antarctic_telemetry.csv"
)

data = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 2. CLEAN DATA
# ============================================================

# Convert required numerical columns to numbers.
# Invalid values become NaN.

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

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# Remove rows where the energy target is missing.

data = data.dropna(
    subset=["energy_consumed_kwh"]
).reset_index(drop=True)


print("\nCleaned dataset:")
print(data.shape)


# ============================================================
# 3. PREPARE DATA FOR EXISTING ML MODEL
# ============================================================

# Your original ML model expects these names.
# We map the Antarctica dataset to those names.

data["Temperature"] = data["temperature_celsius"]

data["Wind_Speed"] = data["wind_speed_knots"]

data["Occupancy"] = data["station_occupancy"]

data["Battery_Level"] = data["battery_level_percent"]

data["Generator_Load"] = data["generator_load_percent"]

data["Fuel_Level"] = data["fuel_level_liters"]


# Your Antarctica dataset does not contain solar radiation.
# Keep the existing model structure unchanged.

data["Solar_Radiation"] = 0


# Energy target

data["Energy_Consumption"] = data["energy_consumed_kwh"]


# Remove any rows with missing model values.

model_columns = [
    "Temperature",
    "Wind_Speed",
    "Solar_Radiation",
    "Occupancy",
    "Battery_Level",
    "Generator_Load",
    "Fuel_Level",
    "Energy_Consumption"
]

data = data.dropna(
    subset=model_columns
).reset_index(drop=True)


# ============================================================
# 4. MODEL FEATURES
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

y = data["energy_consumed_kwh"]


# ============================================================
# 5. TRAIN ENERGY MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


energy_model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)


energy_model.fit(

    X_train,

    y_train

)


# ============================================================
# 6. ENERGY MODEL EVALUATION
# ============================================================

y_pred = energy_model.predict(
    X_test
)


mae = mean_absolute_error(

    y_test,

    y_pred

)


mse = mean_squared_error(

    y_test,

    y_pred

)


r2 = r2_score(

    y_test,

    y_pred

)


print("\n==============================")

print("ENERGY MODEL PERFORMANCE")

print("==============================")


print(

    "MAE:",

    round(mae, 2)

)


print(

    "MSE:",

    round(mse, 2)

)


print(

    "R2 Score:",

    round(r2, 2)

)


# ============================================================
# 7. FEATURE IMPORTANCE
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


print(

    importance.to_string(index=False)

)


# ============================================================
# 8. CURRENT ANTARCTICA STATION CONDITIONS
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
# 9. PREDICT ENERGY CONSUMPTION
# ============================================================

predicted_energy = energy_model.predict(

    current_station[features]

)[0]


print("\n==============================")

print("CURRENT STATION PREDICTION")

print("==============================")


print(

    "Predicted Energy Consumption:",

    round(predicted_energy, 2),

    "kWh"

)


# ============================================================
# 10. REMAINING ENERGY
# ============================================================

battery_capacity = 1000  # kWh


battery_percentage = current_station[

    "Battery_Level"

].iloc[0]


energy_available = (

    battery_percentage / 100

) * battery_capacity


solar_generation = (

    current_station["Solar_Radiation"].iloc[0]

)


total_energy_available = (

    energy_available +

    solar_generation

)


energy_left = (

    total_energy_available -

    predicted_energy

)


# Prevent negative displayed energy.

energy_left = max(

    0,

    energy_left

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
# 11. ENERGY ENDURANCE
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
# 12. REMAINING FUEL
# ============================================================

current_fuel = current_station[

    "Fuel_Level"

].iloc[0]


generator_fuel_rate = 70  # litres/day


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
# 13. FUEL ALERT
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
# 14. ENERGY ALERT
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
# 15. ANOMALY DETECTION
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


# ============================================================
# 16. CURRENT STATION ANOMALY CHECK
# ============================================================

current_energy = predicted_energy


current_anomaly_data = current_station.copy()


current_anomaly_data[

    "Energy_Consumption"

] = current_energy


anomaly_result = anomaly_model.predict(

    current_anomaly_data[

        anomaly_features

    ]

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
# 17. RISK SCORE
# ============================================================

risk_score = 0


if fuel_days_remaining < 7:

    risk_score += 30


if days_of_energy < 5:

    risk_score += 30


if anomaly_status == "ANOMALY":

    risk_score += 40


risk_score = min(

    risk_score,

    100

)


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
# 18. NEXT VISIT RESOURCE RECOMMENDATION
# ============================================================

planning_days = 30


fuel_required = (

    generator_fuel_rate *

    planning_days

)


fuel_reserve = (

    fuel_required * 0.20

)


total_fuel_required = (

    fuel_required +

    fuel_reserve

)


fuel_to_bring = max(

    0,

    total_fuel_required -

    current_fuel

)


print("\n==============================")

print("NEXT ANTARCTICA VISIT PLAN")

print("==============================")


print(

    "Recommended Fuel:",

    round(fuel_to_bring, 2),

    "litres"

)


# ============================================================
# 19. RESOURCE RECOMMENDATION
# ============================================================

if energy_status == "CRITICAL":

    energy_recommendation = (

        "Additional battery capacity"

    )

elif energy_status == "WARNING":

    energy_recommendation = (

        "Inspect battery and solar system"

    )

else:

    energy_recommendation = (

        "No immediate energy action"

    )


print(

    "Energy Recommendation:",

    energy_recommendation

)


# ============================================================
# 20. MAINTENANCE RECOMMENDATION
# ============================================================

if anomaly_status == "ANOMALY":

    maintenance_recommendation = (

        "Generator/equipment inspection"

    )

else:

    maintenance_recommendation = (

        "Routine inspection"

    )


print(

    "Maintenance Recommendation:",

    maintenance_recommendation

)


# ============================================================
# 21. FINAL DIGITAL TWIN REPORT
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


print(

    "Energy Recommendation:",

    energy_recommendation

)


print(

    "Maintenance Recommendation:",

    maintenance_recommendation

)


print("==========================================")


# ============================================================
# 22. SAVE OUTPUT CSV
# ============================================================

# Create a complete output record containing:
# Original Antarctica data
# + model-compatible columns
# + prediction
# + digital twin calculations


output_data = data.copy()


output_data["Predicted_Energy_Consumption"] = (

    energy_model.predict(

        output_data[features]

    )

)


output_data["Anomaly_Status"] = np.where(

    anomaly_model.predict(

        output_data[anomaly_features]

    ) == -1,

    "ANOMALY",

    "NORMAL"

)


output_data["Energy_Endurance_Days"] = (

    output_data["battery_level_percent"] / 100

    * battery_capacity

    / output_data["Predicted_Energy_Consumption"].replace(

        0,

        np.nan

    )

)


output_data["Fuel_Endurance_Days"] = (

    output_data["fuel_level_liters"] /

    generator_fuel_rate

)


output_data["Risk_Score"] = 0


output_data.loc[

    output_data["Fuel_Endurance_Days"] < 7,

    "Risk_Score"

] += 30


output_data.loc[

    output_data["Energy_Endurance_Days"] < 5,

    "Risk_Score"

] += 30


output_data.loc[

    output_data["Anomaly_Status"] == "ANOMALY",

    "Risk_Score"

] += 40


output_data["Risk_Score"] = (

    output_data["Risk_Score"].clip(

        0,

        100

    )

)


output_data["Recommended_Fuel_Litres"] = (

    np.maximum(

        0,

        total_fuel_required -

        output_data["fuel_level_liters"]

    )

)


output_data["Next_Visit_Required"] = np.where(

    (output_data["Fuel_Endurance_Days"] < 3) |

    (output_data["Energy_Endurance_Days"] < 2) |

    (output_data["Risk_Score"] >= 70),

    "YES",

    "NO"

)


output_data["Resource_Recommendation"] = np.where(

    output_data["Risk_Score"] >= 70,

    "Fuel + Battery + Maintenance",

    np.where(

        output_data["Fuel_Endurance_Days"] < 7,

        "Fuel Refill",

        "No Immediate Resource Required"

    )

)


# Save output inside the repository.

output_file = os.path.join(

    BASE_DIR,

    "antarctica_digital_twin_predictions.csv"

)


output_data.to_csv(

    output_file,

    index=False

)


print(

    "\n✅ Complete prediction file saved:"

)

print(output_file)


# ============================================================
# 23. SAVE CURRENT DIGITAL TWIN RESULT TO MYSQL
# ============================================================

import mysql.connector


try:

    db = mysql.connector.connect(

        host="localhost",

        user="root",

        password="lavanya",

        database="antarctica"

    )


    cursor = db.cursor()


    print("\nConnected to MySQL!")


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

        float(

            current_station["Temperature"].iloc[0]

        ),

        float(

            current_station["Wind_Speed"].iloc[0]

        ),

        float(

            current_station["Solar_Radiation"].iloc[0]

        ),

        int(

            current_station["Occupancy"].iloc[0]

        ),

        float(

            current_station["Battery_Level"].iloc[0]

        ),

        float(

            current_station["Generator_Load"].iloc[0]

        ),

        float(

            current_station["Fuel_Level"].iloc[0]

        ),

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


    cursor.execute(

        sql,

        values

    )


    db.commit()


    print(

        "✅ Digital Twin prediction saved to MySQL!"

    )


    cursor.close()

    db.close()


except mysql.connector.Error as error:

    print(

        "⚠️ MySQL Error:",

        error

    )