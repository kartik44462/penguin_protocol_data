import os
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mysql.connector
from mysql.connector import Error

warnings.filterwarnings("ignore")


# ============================================================
# ANTARCTICA DIGITAL TWIN - REFINED ML PIPELINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "MODEL_TRAINING_14COL.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Antarctic Digital Twin Prediction.csv")

# -------------------------
# MySQL configuration
# -------------------------
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "lavanya",
    "database": "antarctica",
}

# -------------------------
# Station / planning settings
# -------------------------
BATTERY_CAPACITY_KWH = 1000.0
PLANNING_DAYS = 30
FUEL_RESERVE = 0.20

# Set this to False if you want to use the latest CSV row
# as the "current station condition" automatically.
USE_DEMO_CURRENT_CONDITIONS = True

# Select which station the current-condition prediction represents.
# Valid values: MAITRI or BHARATI
DEMO_STATION_ID = "MAITRI"

DEMO_CURRENT_CONDITIONS = {
    "Temperature": -25.0,
    "Wind_Speed": 30.0,
    "Solar_Radiation": 150.0,
    "Occupancy": 25.0,
    "Battery_Level": 65.0,
    "Generator_Load": 60.0,
    "Fuel_Level": 500.0,
}


# ============================================================
# 1. HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def require_columns(dataframe, columns, name="dataset"):
    missing = [c for c in columns if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {name}: {missing}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )


# ============================================================
# 2. LOAD DATA
# ============================================================

print_header("ANTARCTICA DIGITAL TWIN - DATA LOADING")

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    raise SystemExit(
        f"\nERROR: {os.path.basename(INPUT_FILE)} was not found.\n"
        f"Place MODEL_TRAINING_14COL.csv in:\n{BASE_DIR}"
    )

print("\nDataset loaded successfully!")
print(f"Input file: {INPUT_FILE}")
print(f"Records: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 3. NORMALIZE COLUMN NAMES
# ============================================================
# Your CSV uses names such as Temperature and Wind_Speed,
# while the previous code expected names such as
# temperature_celsius and wind_speed_knots.
#
# We standardize them here so the rest of the pipeline is
# independent of the CSV naming convention.

print_header("DATA CLEANING")

column_mapping = {
    "Temperature": "temperature_celsius",
    "Wind_Speed": "wind_speed_knots",
    "Pressure": "pressure_hpa",
    "Humidity": "humidity_percent",
    "Solar_Radiation": "solar_radiation_wm2",
    "Generator_Load": "generator_load_percent",
    "Energy_consumption": "energy_consumed_kwh",
    "Battery_Level": "battery_level_percent",
    "Fuel_Level": "fuel_level_liters",
    "Fuel_Burn_Rate": "fuel_burn_rate_lph",
    "Food_Inventory": "food_inventory_kg",
    "Station_Occupancy": "station_occupancy",
}

df = df.rename(columns=column_mapping)

required_columns = [
    "temperature_celsius",
    "wind_speed_knots",
    "pressure_hpa",
    "humidity_percent",
    "solar_radiation_wm2",
    "generator_load_percent",
    "energy_consumed_kwh",
    "battery_level_percent",
    "fuel_level_liters",
    "fuel_burn_rate_lph",
    "food_inventory_kg",
    "station_occupancy",
]

require_columns(df, required_columns)

df = df.dropna(how="all").copy()

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns).copy()

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------------------
# STATION IDENTIFICATION
# ------------------------------------------------------------
# station_id is now an actual ML feature, not just a database field.
if "station_id" not in df.columns:
    raise ValueError(
        "station_id is required so the model can differentiate "
        "MAITRI and BHARATI."
    )

df["station_id"] = (
    df["station_id"]
    .astype(str)
    .str.strip()
    .str.upper()
)

valid_stations = {"MAITRI", "BHARATI"}
unknown_stations = sorted(set(df["station_id"]) - valid_stations)

if unknown_stations:
    raise ValueError(
        f"Unknown station_id values found: {unknown_stations}. "
        f"Use only: {sorted(valid_stations)}"
    )

print("Cleaned dataset successfully!")
print(f"Valid records: {len(df)}")
print("\nStation distribution:")
print(df["station_id"].value_counts().to_string())


# ============================================================
# 4. PREPARE ML FEATURES
# ============================================================

print_header("PREPARING ML FEATURES")

# Encode station identity numerically for the Random Forest:
# MAITRI = 0, BHARATI = 1
df["Station_Code"] = df["station_id"].map({
    "MAITRI": 0,
    "BHARATI": 1,
}).astype(int)

model_df = pd.DataFrame({
    "Temperature": df["temperature_celsius"],
    "Wind_Speed": df["wind_speed_knots"],
    "Solar_Radiation": df["solar_radiation_wm2"],
    "Occupancy": df["station_occupancy"],
    "Battery_Level": df["battery_level_percent"],
    "Generator_Load": df["generator_load_percent"],
    "Fuel_Level": df["fuel_level_liters"],
    "Station_Code": df["Station_Code"],
})

target = df["energy_consumed_kwh"]

features = model_df.columns.tolist()

print("\nFeatures used:")
for feature in features:
    print(f"  - {feature}")

print("\nTarget:")
print("  Energy_consumption (kWh)")


# ============================================================
# 5. TIME-AWARE TRAIN / TEST SPLIT
# ============================================================
# The old code used a random split. For timestamped station data,
# a chronological split is more realistic: train on earlier data
# and test on later data.

print_header("MODEL TRAINING / TEST SPLIT")

split_index = int(len(model_df) * 0.80)

if split_index <= 0 or split_index >= len(model_df):
    raise ValueError("Dataset is too small for an 80/20 train-test split.")

X_train = model_df.iloc[:split_index]
X_test = model_df.iloc[split_index:]
y_train = target.iloc[:split_index]
y_test = target.iloc[split_index:]

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ============================================================
# 6. RANDOM FOREST ENERGY MODEL
# ============================================================

print_header("ENERGY PREDICTION MODEL")

energy_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

energy_model.fit(X_train, y_train)

y_pred = energy_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEnergy Model Performance")
print("-" * 40)
print(f"MAE  : {mae:.4f} kWh")
print(f"RMSE : {rmse:.4f} kWh")
print(f"R²   : {r2:.4f}")


# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

print_header("FEATURE IMPORTANCE")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": energy_model.feature_importances_,
}).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))

print("\nStation-aware model check:")
for station in ["MAITRI", "BHARATI"]:
    station_rows = int((df["station_id"] == station).sum())
    print(f"  {station}: {station_rows} records")

if not all((df["station_id"] == s).any() for s in ["MAITRI", "BHARATI"]):
    print(
        "\nWARNING: The dataset does not contain both MAITRI and BHARATI. "
        "The model can accept both station codes, but it can only learn "
        "station-specific behavior from stations represented in training data."
    )


# ============================================================
# 8. PREDICT ENERGY FOR ALL DATA
# ============================================================

df["predicted_energy_kwh"] = energy_model.predict(model_df)


# ============================================================
# 9. DETERMINE DATA INTERVAL
# ============================================================

interval_hours = 5 / 60  # default = 5 minutes

if "timestamp" in df.columns and df["timestamp"].notna().sum() >= 2:
    time_diff_hours = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .dropna()
    )

    positive_intervals = time_diff_hours[time_diff_hours > 0]

    if not positive_intervals.empty:
        interval_hours = float(positive_intervals.median())

interval_days = interval_hours / 24

print(f"\nDetected data interval: {interval_hours * 60:.2f} minutes")


# ============================================================
# 10. ISOLATION FOREST ANOMALY DETECTION
# ============================================================

print_header("ANOMALY DETECTION")

anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42,
    n_jobs=-1,
)

anomaly_model.fit(model_df)

df["anomaly_prediction"] = anomaly_model.predict(model_df)

df["anomaly_status"] = np.where(
    df["anomaly_prediction"] == -1,
    "ANOMALY",
    "NORMAL",
)

anomaly_count = int((df["anomaly_status"] == "ANOMALY").sum())

print(f"Anomalies detected: {anomaly_count}")
print(f"Normal records   : {len(df) - anomaly_count}")


# ============================================================
# 11. CURRENT STATION CONDITIONS
# ============================================================

print_header("CURRENT STATION PREDICTION")

if USE_DEMO_CURRENT_CONDITIONS:
    DEMO_STATION_ID = str(DEMO_STATION_ID).strip().upper()

    if DEMO_STATION_ID not in {"MAITRI", "BHARATI"}:
        raise ValueError(
            "DEMO_STATION_ID must be either 'MAITRI' or 'BHARATI'."
        )

    current_station = pd.DataFrame([{
        **DEMO_CURRENT_CONDITIONS,
        "Station_Code": 0 if DEMO_STATION_ID == "MAITRI" else 1,
    }])
    current_station_id = DEMO_STATION_ID
    current_source = (
        f"Demo/current operating conditions for {current_station_id}"
    )
else:
    # Use the latest record from each station independently, then select
    # the latest overall station record as the current station condition.
    latest = df.dropna(subset=["timestamp"]).iloc[-1]

    current_station_id = str(latest["station_id"]).upper()

    current_station = pd.DataFrame([{
        "Temperature": latest["temperature_celsius"],
        "Wind_Speed": latest["wind_speed_knots"],
        "Solar_Radiation": latest["solar_radiation_wm2"],
        "Occupancy": latest["station_occupancy"],
        "Battery_Level": latest["battery_level_percent"],
        "Generator_Load": latest["generator_load_percent"],
        "Fuel_Level": latest["fuel_level_liters"],
        "Station_Code": int(latest["Station_Code"]),
    }])
    current_source = (
        f"Latest record from {current_station_id}"
    )

print(f"Current station: {current_station_id}")
print(f"Current condition source: {current_source}")

current_energy_prediction = float(
    energy_model.predict(current_station)[0]
)

print(
    f"\nPredicted Energy Consumption per "
    f"{interval_hours * 60:.0f}-minute interval: "
    f"{current_energy_prediction:.2f} kWh"
)


# ============================================================
# 12. ENERGY CALCULATIONS
# ============================================================

battery_level_percent = float(current_station["Battery_Level"].iloc[0])
current_battery = (
    battery_level_percent / 100.0
) * BATTERY_CAPACITY_KWH

# Convert interval energy to an approximate daily requirement.
energy_consumption_per_day = (
    current_energy_prediction / interval_hours * 24
)

if energy_consumption_per_day > 0:
    energy_endurance_days = current_battery / energy_consumption_per_day
else:
    energy_endurance_days = float("inf")

# Battery status is based on the available battery percentage.
if battery_level_percent <= 20:
    energy_status = "CRITICAL"
elif battery_level_percent <= 40:
    energy_status = "WARNING"
else:
    energy_status = "NORMAL"

print(f"\nBattery Energy Remaining: {current_battery:.2f} kWh")
print(
    f"Estimated Daily Energy Consumption: "
    f"{energy_consumption_per_day:.2f} kWh/day"
)
print(
    f"Energy Endurance: "
    f"{energy_endurance_days:.2f} days"
)
print("Energy Status:", energy_status)


# ============================================================
# 13. FUEL CALCULATIONS
# ============================================================

current_fuel = float(current_station["Fuel_Level"].iloc[0])

# Use the selected station's historical median fuel burn rate.
# This prevents MAITRI's fuel behavior from being mixed with BHARATI's.
station_fuel_history = df[
    df["station_id"] == current_station_id
]["fuel_burn_rate_lph"]

if station_fuel_history.dropna().empty:
    # Fallback only if the selected station has no valid fuel-rate history.
    fuel_burn_rate_lph = float(df["fuel_burn_rate_lph"].median())
    fuel_rate_source = "All-station fallback"
else:
    fuel_burn_rate_lph = float(station_fuel_history.median())
    fuel_rate_source = f"{current_station_id} historical median"

print(f"Fuel-rate source: {fuel_rate_source}")

if fuel_burn_rate_lph > 0:
    fuel_consumption_per_day = fuel_burn_rate_lph * 24
    fuel_endurance_days = current_fuel / fuel_consumption_per_day
else:
    fuel_consumption_per_day = 0.0
    fuel_endurance_days = float("inf")

if fuel_endurance_days < 3:
    fuel_status = "CRITICAL"
elif fuel_endurance_days < 7:
    fuel_status = "WARNING"
else:
    fuel_status = "NORMAL"

print_header("FUEL STATUS")
print(f"Fuel Remaining: {current_fuel:.2f} litres")
print(f"Fuel Burn Rate: {fuel_burn_rate_lph:.2f} L/hour")
print(f"Fuel Consumption: {fuel_consumption_per_day:.2f} L/day")
print(f"Fuel Endurance: {fuel_endurance_days:.2f} days")
print("Fuel Status:", fuel_status)


# ============================================================
# 14. ALERTS
# ============================================================

fuel_alert = (
    "FUEL SUPPLY REQUIRED"
    if fuel_endurance_days < 7
    else "FUEL LEVEL SUFFICIENT"
)

energy_alert = (
    "ENERGY RESERVE LOW"
    if energy_endurance_days < 5
    else "ENERGY LEVEL SUFFICIENT"
)

print("\nFuel Alert:", fuel_alert)
print("Energy Alert:", energy_alert)


# ============================================================
# 15. CURRENT ANOMALY
# ============================================================

current_anomaly_result = int(
    anomaly_model.predict(current_station)[0]
)

current_anomaly_status = (
    "ANOMALY"
    if current_anomaly_result == -1
    else "NORMAL"
)

print("\nCurrent Anomaly Status:", current_anomaly_status)


# ============================================================
# 16. STATION RISK SCORE
# ============================================================

risk_score = 0

if fuel_endurance_days < 7:
    risk_score += 30

if energy_endurance_days < 5:
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

print_header("STATION RISK")
print(f"Risk Score : {risk_score} / 100")
print(f"Risk Status: {risk_status}")


# ============================================================
# 17. NEXT ANTARCTICA VISIT PLAN
# ============================================================

# Required fuel for the selected planning period, with reserve.
required_fuel = (
    PLANNING_DAYS
    * fuel_consumption_per_day
    * (1 + FUEL_RESERVE)
)

additional_fuel = max(0.0, required_fuel - current_fuel)

print_header("NEXT ANTARCTICA VISIT PLAN")
print(f"Planning Period: {PLANNING_DAYS} days")
print(f"Fuel Reserve: {FUEL_RESERVE * 100:.0f}%")
print(f"Recommended Fuel: {required_fuel:.2f} litres")
print(f"Additional Fuel Required: {additional_fuel:.2f} litres")


# ============================================================
# 18. RECOMMENDATIONS
# ============================================================

if energy_status == "CRITICAL":
    energy_recommendation = (
        "Immediately reduce non-essential energy consumption "
        "and prioritize critical station loads."
    )
elif energy_status == "WARNING":
    energy_recommendation = (
        "Monitor battery usage and reduce unnecessary loads."
    )
else:
    energy_recommendation = (
        "Energy availability is currently within acceptable limits."
    )

if current_anomaly_status == "ANOMALY":
    maintenance_recommendation = (
        "Immediate equipment inspection recommended."
    )
elif risk_score >= 40:
    maintenance_recommendation = (
        "Schedule preventive maintenance before the next visit."
    )
else:
    maintenance_recommendation = (
        "Continue routine preventive maintenance."
    )

print("\nEnergy Recommendation:")
print(energy_recommendation)

print("\nMaintenance Recommendation:")
print(maintenance_recommendation)


# ============================================================
# 19. SAVE CALCULATED VALUES
# ============================================================

# These are station-level current-state values, so they are
# repeated in the exported dataset for easy dashboard/database use.

df["energy_remaining_kwh"] = current_battery
df["energy_consumption_per_day_kwh"] = energy_consumption_per_day
df["energy_endurance_days"] = energy_endurance_days
df["fuel_consumption_per_day_litres"] = fuel_consumption_per_day
df["fuel_endurance_days"] = fuel_endurance_days
df["risk_score"] = risk_score
df["risk_status"] = risk_status
df["recommended_fuel_litres"] = required_fuel
df["additional_fuel_required_litres"] = additional_fuel
df["energy_status"] = energy_status
df["fuel_status"] = fuel_status


# ============================================================
# 20. FINAL DIGITAL TWIN REPORT
# ============================================================

print_header("ANTARCTICA DIGITAL TWIN REPORT")

station_name = current_station_id

print(f"Station: {station_name}")
print(f"Predicted Energy / Interval: {current_energy_prediction:.2f} kWh")
print(f"Daily Energy Requirement: {energy_consumption_per_day:.2f} kWh/day")
print(f"Energy Remaining: {current_battery:.2f} kWh")
print(f"Energy Endurance: {energy_endurance_days:.2f} days")
print(f"Fuel Remaining: {current_fuel:.2f} litres")
print(f"Fuel Endurance: {fuel_endurance_days:.2f} days")
print(f"Anomaly Status: {current_anomaly_status}")
print(f"Risk Score: {risk_score} / 100")
print(f"Risk Status: {risk_status}")
print(f"Recommended Fuel: {required_fuel:.2f} litres")
print(f"Additional Fuel Required: {additional_fuel:.2f} litres")
print(f"Energy Status: {energy_status}")
print(f"Fuel Status: {fuel_status}")

print("=" * 60)


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nPrediction file saved successfully!")
print(f"Output: {OUTPUT_FILE}")

print("\nStation-wise prediction summary:")
station_summary = (
    df.groupby("station_id")["predicted_energy_kwh"]
    .agg(["count", "mean", "min", "max"])
)
print(station_summary.to_string())

# ============================================================
# 22. MYSQL DATABASE - UPLOAD ALL PREDICTIONS
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

    # --------------------------------------------------------
    # EXACT COLUMNS TO UPLOAD
    # --------------------------------------------------------

    mysql_columns = [
        "timestamp",
        "station_id",
        "temperature_celsius",
        "wind_speed_knots",
        "pressure_hpa",
        "humidity_percent",
        "solar_radiation_wm2",
        "generator_load_percent",
        "energy_consumed_kwh",
        "battery_level_percent",
        "fuel_level_liters",
        "fuel_burn_rate_lph",
        "food_inventory_kg",
        "station_occupancy",
        "predicted_energy_kwh",
        "anomaly_prediction",
        "anomaly_status",
        "energy_remaining_kwh",
        "energy_consumption_per_day_kwh",
        "energy_endurance_days",
        "fuel_consumption_per_day_litres",
        "fuel_endurance_days",
        "risk_score",
        "risk_status",
        "recommended_fuel_litres",
        "additional_fuel_required_litres",
        "energy_status",
        "fuel_status"
    ]

    # --------------------------------------------------------
    # CHECK THAT ALL COLUMNS EXIST
    # --------------------------------------------------------

    missing_columns = [
        column for column in mysql_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataframe: {missing_columns}"
        )

    print("\nAll 28 columns found successfully!")

    # --------------------------------------------------------
    # INSERT QUERY
    # --------------------------------------------------------

    insert_query = """
    INSERT INTO predictions (
        timestamp,
        station_id,
        temperature_celsius,
        wind_speed_knots,
        pressure_hpa,
        humidity_percent,
        solar_radiation_wm2,
        generator_load_percent,
        energy_consumed_kwh,
        battery_level_percent,
        fuel_level_liters,
        fuel_burn_rate_lph,
        food_inventory_kg,
        station_occupancy,
        predicted_energy_kwh,
        anomaly_prediction,
        anomaly_status,
        energy_remaining_kwh,
        energy_consumption_per_day_kwh,
        energy_endurance_days,
        fuel_consumption_per_day_litres,
        fuel_endurance_days,
        risk_score,
        risk_status,
        recommended_fuel_litres,
        additional_fuel_required_litres,
        energy_status,
        fuel_status
    )
  VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s
)
    """

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    upload_df = df[mysql_columns].copy()

    # Convert pandas NaN values to Python None
    # so MySQL receives NULL only when the CSV actually
    # contains a missing value.
    upload_df = upload_df.where(
        pd.notnull(upload_df),
        None
    )

    values = [
        tuple(row)
        for row in upload_df.itertuples(
            index=False,
            name=None
        )
    ]

    print(f"\nRows ready for upload: {len(values)}")

    # --------------------------------------------------------
    # UPLOAD ALL ROWS
    # --------------------------------------------------------

    cursor.executemany(
        insert_query,
        values
    )

    connection.commit()

    print("\n" + "=" * 60)
    print("MYSQL UPLOAD SUCCESSFUL")
    print("=" * 60)

    print(f"Rows uploaded: {cursor.rowcount}")

    # --------------------------------------------------------
    # VERIFY UPLOAD
    # --------------------------------------------------------

    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total_rows = cursor.fetchone()[0]

    print(f"Total rows currently in MySQL: {total_rows}")

    cursor.close()
    connection.close()

    print("MySQL connection closed.")

except mysql.connector.Error as error:

    print("\nMYSQL ERROR:")
    print(error)

except Exception as error:

    print("\nUPLOAD ERROR:")
    print(error)


# ============================================================
# 23. COMPLETED
# ============================================================

print_header("DIGITAL TWIN PROCESS COMPLETED")

# ============================================================
# 24. REAL-TIME MAITRI + BHARATI DIGITAL TWIN STREAM
# ============================================================

import time
from datetime import datetime, timedelta

RUN_REALTIME = True

# Every 5 real seconds, generate one new reading for both stations.
REALTIME_INTERVAL_SECONDS = 5

# Simulation speed:
# 1 real tick = 60 simulated minutes.
SIMULATED_MINUTES_PER_TICK = 60

STATIONS = ["MAITRI", "BHARATI"]

# These are SIMULATION starting values.
# Replace generate_realtime_conditions() with actual sensor/API/MQTT
# input later if real Antarctic sensor data becomes available.
REALTIME_BASE_CONDITIONS = {
    "MAITRI": {
        "Temperature": -25.0,
        "Wind_Speed": 30.0,
        "Solar_Radiation": 150.0,
        "Occupancy": 25.0,
        "Battery_Level": 65.0,
        "Generator_Load": 60.0,
        "Fuel_Level": 500.0,
    },
    "BHARATI": {
        "Temperature": -18.0,
        "Wind_Speed": 25.0,
        "Solar_Radiation": 180.0,
        "Occupancy": 20.0,
        "Battery_Level": 70.0,
        "Generator_Load": 55.0,
        "Fuel_Level": 550.0,
    },
}


def generate_realtime_conditions(station_id, previous):
    """Generate one simulated sensor reading for a station."""
    values = previous.copy()

    values["Temperature"] += np.random.normal(0, 0.4)
    values["Wind_Speed"] = max(
        0.0, values["Wind_Speed"] + np.random.normal(0, 2.0)
    )
    values["Solar_Radiation"] = max(
        0.0, values["Solar_Radiation"] + np.random.normal(0, 15.0)
    )
    values["Occupancy"] = max(
        0.0, values["Occupancy"] + np.random.normal(0, 0.5)
    )
    values["Battery_Level"] = np.clip(
        values["Battery_Level"] + np.random.normal(0, 1.0),
        0.0, 100.0
    )
    values["Generator_Load"] = np.clip(
        values["Generator_Load"] + np.random.normal(0, 2.0),
        0.0, 100.0
    )
    values["Fuel_Level"] = max(
        0.0,
        values["Fuel_Level"] - np.random.uniform(0.05, 0.25)
    )

    return values


def build_realtime_features(station_id, conditions):
    """Build exactly the feature columns expected by the ML models."""
    return pd.DataFrame([{
        "Temperature": conditions["Temperature"],
        "Wind_Speed": conditions["Wind_Speed"],
        "Solar_Radiation": conditions["Solar_Radiation"],
        "Occupancy": conditions["Occupancy"],
        "Battery_Level": conditions["Battery_Level"],
        "Generator_Load": conditions["Generator_Load"],
        "Fuel_Level": conditions["Fuel_Level"],
        "Station_Code": 0 if station_id == "MAITRI" else 1,
    }])


def ensure_realtime_table(connection):
    """Create a separate table for continuously changing station state."""
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS realtime_station_state (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            station_id VARCHAR(20) NOT NULL,
            temperature_celsius DOUBLE,
            wind_speed_knots DOUBLE,
            solar_radiation_wm2 DOUBLE,
            station_occupancy DOUBLE,
            battery_level_percent DOUBLE,
            generator_load_percent DOUBLE,
            fuel_level_liters DOUBLE,
            predicted_energy_kwh DOUBLE,
            anomaly_prediction INT,
            anomaly_status VARCHAR(20),
            energy_remaining_kwh DOUBLE,
            energy_consumption_per_day_kwh DOUBLE,
            energy_endurance_days DOUBLE,
            fuel_consumption_per_day_litres DOUBLE,
            fuel_endurance_days DOUBLE,
            risk_score INT,
            risk_status VARCHAR(30),
            recommended_fuel_litres DOUBLE,
            additional_fuel_required_litres DOUBLE,
            energy_status VARCHAR(20),
            fuel_status VARCHAR(20)
        )
    """)

    connection.commit()
    cursor.close()


def insert_realtime_state(connection, state):
    cursor = connection.cursor()

    query = """
        INSERT INTO realtime_station_state (
            timestamp, station_id,
            temperature_celsius, wind_speed_knots,
            solar_radiation_wm2, station_occupancy,
            battery_level_percent, generator_load_percent,
            fuel_level_liters, predicted_energy_kwh,
            anomaly_prediction, anomaly_status,
            energy_remaining_kwh, energy_consumption_per_day_kwh,
            energy_endurance_days, fuel_consumption_per_day_litres,
            fuel_endurance_days, risk_score, risk_status,
            recommended_fuel_litres, additional_fuel_required_litres,
            energy_status, fuel_status
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        )
    """

    cursor.execute(query, (
        state["timestamp"],
        state["station_id"],
        state["temperature_celsius"],
        state["wind_speed_knots"],
        state["solar_radiation_wm2"],
        state["station_occupancy"],
        state["battery_level_percent"],
        state["generator_load_percent"],
        state["fuel_level_liters"],
        state["predicted_energy_kwh"],
        state["anomaly_prediction"],
        state["anomaly_status"],
        state["energy_remaining_kwh"],
        state["energy_consumption_per_day_kwh"],
        state["energy_endurance_days"],
        state["fuel_consumption_per_day_litres"],
        state["fuel_endurance_days"],
        state["risk_score"],
        state["risk_status"],
        state["recommended_fuel_litres"],
        state["additional_fuel_required_litres"],
        state["energy_status"],
        state["fuel_status"],
    ))

    connection.commit()
    cursor.close()


def calculate_realtime_state(station_id, conditions, simulated_timestamp):
    """Run ML predictions and calculations for one station."""
    features_now = build_realtime_features(station_id, conditions)

    predicted_energy = float(
        energy_model.predict(features_now)[0]
    )

    anomaly_result = int(
        anomaly_model.predict(features_now)[0]
    )

    anomaly_status = (
        "ANOMALY" if anomaly_result == -1 else "NORMAL"
    )

    battery_percent = float(conditions["Battery_Level"])
    current_battery = (
        battery_percent / 100.0
    ) * BATTERY_CAPACITY_KWH

    daily_energy = (
        predicted_energy / interval_hours * 24
        if interval_hours > 0 else 0.0
    )

    energy_endurance = (
        current_battery / daily_energy
        if daily_energy > 0 else float("inf")
    )

    if battery_percent <= 20:
        energy_status = "CRITICAL"
    elif battery_percent <= 40:
        energy_status = "WARNING"
    else:
        energy_status = "NORMAL"

    # Use this station's historical fuel-burn rate.
    station_fuel = df[
        df["station_id"] == station_id
    ]["fuel_burn_rate_lph"].dropna()

    if station_fuel.empty:
        burn_rate = float(df["fuel_burn_rate_lph"].median())
    else:
        burn_rate = float(station_fuel.median())

    current_fuel = float(conditions["Fuel_Level"])
    fuel_per_day = burn_rate * 24 if burn_rate > 0 else 0.0

    fuel_endurance = (
        current_fuel / fuel_per_day
        if fuel_per_day > 0 else float("inf")
    )

    if fuel_endurance < 3:
        fuel_status = "CRITICAL"
    elif fuel_endurance < 7:
        fuel_status = "WARNING"
    else:
        fuel_status = "NORMAL"

    risk_score = 0

    if fuel_endurance < 7:
        risk_score += 30

    if energy_endurance < 5:
        risk_score += 30

    if anomaly_status == "ANOMALY":
        risk_score += 40

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_status = "HIGH RISK"
    elif risk_score >= 40:
        risk_status = "MEDIUM RISK"
    else:
        risk_status = "LOW RISK"

    required_fuel = (
        PLANNING_DAYS
        * fuel_per_day
        * (1 + FUEL_RESERVE)
    )

    additional_fuel = max(
        0.0,
        required_fuel - current_fuel
    )

    return {
        "timestamp": simulated_timestamp,
        "station_id": station_id,
        "temperature_celsius": float(conditions["Temperature"]),
        "wind_speed_knots": float(conditions["Wind_Speed"]),
        "solar_radiation_wm2": float(conditions["Solar_Radiation"]),
        "station_occupancy": float(conditions["Occupancy"]),
        "battery_level_percent": battery_percent,
        "generator_load_percent": float(conditions["Generator_Load"]),
        "fuel_level_liters": current_fuel,
        "predicted_energy_kwh": predicted_energy,
        "anomaly_prediction": anomaly_result,
        "anomaly_status": anomaly_status,
        "energy_remaining_kwh": current_battery,
        "energy_consumption_per_day_kwh": daily_energy,
        "energy_endurance_days": energy_endurance,
        "fuel_consumption_per_day_litres": fuel_per_day,
        "fuel_endurance_days": fuel_endurance,
        "risk_score": risk_score,
        "risk_status": risk_status,
        "recommended_fuel_litres": required_fuel,
        "additional_fuel_required_litres": additional_fuel,
        "energy_status": energy_status,
        "fuel_status": fuel_status,
    }


def run_realtime_pipeline():
    """Continuously update both Maitri and Bharati."""
    print_header("REAL-TIME MAITRI + BHARATI DIGITAL TWIN")

    connection = None

    try:
        connection = mysql.connector.connect(
            host=MYSQL_CONFIG["host"],
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            database=MYSQL_CONFIG["database"],
        )

        ensure_realtime_table(connection)

        live_conditions = {
            station: REALTIME_BASE_CONDITIONS[station].copy()
            for station in STATIONS
        }

        if "timestamp" in df.columns and df["timestamp"].notna().any():
            simulated_time = (
                df["timestamp"].dropna().max().to_pydatetime()
            )
        else:
            simulated_time = datetime.now()

        print("\nREAL-TIME PIPELINE STARTED")
        print(f"Update interval: {REALTIME_INTERVAL_SECONDS} seconds")
        print(
            f"Simulation speed: "
            f"{SIMULATED_MINUTES_PER_TICK} simulated minutes/tick"
        )
        print("Stations: MAITRI + BHARATI")
        print("Press Ctrl+C to stop.\n")

        while True:
            simulated_time += timedelta(
                minutes=SIMULATED_MINUTES_PER_TICK
            )

            for station in STATIONS:

                # Generate the next live reading.
                live_conditions[station] = (
                    generate_realtime_conditions(
                        station,
                        live_conditions[station]
                    )
                )

                # Run station-specific ML prediction.
                state = calculate_realtime_state(
                    station,
                    live_conditions[station],
                    simulated_time
                )

                # Save live state to MySQL.
                insert_realtime_state(connection, state)

                print(
                    f"[{state['timestamp']}] "
                    f"{station:7s} | "
                    f"Energy={state['predicted_energy_kwh']:.2f} kWh | "
                    f"Battery={state['battery_level_percent']:.1f}% | "
                    f"Fuel={state['fuel_level_liters']:.1f} L | "
                    f"Anomaly={state['anomaly_status']} | "
                    f"Risk={state['risk_score']}/100"
                )

            time.sleep(REALTIME_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nReal-time pipeline stopped.")

    except mysql.connector.Error as error:
        print("\nREAL-TIME MYSQL ERROR:")
        print(error)

    finally:
        if connection is not None:
            try:
                connection.close()
                print("MySQL connection closed.")
            except Exception:
                pass


if RUN_REALTIME:
    run_realtime_pipeline()


# ============================================================
# 25. COMPLETED
# ============================================================
