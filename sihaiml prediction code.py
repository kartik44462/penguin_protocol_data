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

print("Cleaned dataset successfully!")
print(f"Valid records: {len(df)}")


# ============================================================
# 4. PREPARE ML FEATURES
# ============================================================

print_header("PREPARING ML FEATURES")

model_df = pd.DataFrame({
    "Temperature": df["temperature_celsius"],
    "Wind_Speed": df["wind_speed_knots"],
    "Solar_Radiation": df["solar_radiation_wm2"],
    "Occupancy": df["station_occupancy"],
    "Battery_Level": df["battery_level_percent"],
    "Generator_Load": df["generator_load_percent"],
    "Fuel_Level": df["fuel_level_liters"],
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
    current_station = pd.DataFrame([DEMO_CURRENT_CONDITIONS])
    current_source = "Demo/current operating conditions"
else:
    latest = df.iloc[-1]
    current_station = pd.DataFrame([{
        "Temperature": latest["temperature_celsius"],
        "Wind_Speed": latest["wind_speed_knots"],
        "Solar_Radiation": latest["solar_radiation_wm2"],
        "Occupancy": latest["station_occupancy"],
        "Battery_Level": latest["battery_level_percent"],
        "Generator_Load": latest["generator_load_percent"],
        "Fuel_Level": latest["fuel_level_liters"],
    }])
    current_source = "Latest record from training dataset"

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

# Use the dataset's historical median fuel burn rate instead of
# hard-coding 70 litres/day.
fuel_burn_rate_lph = float(df["fuel_burn_rate_lph"].median())

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

station_name = (
    str(df["station_id"].iloc[-1])
    if "station_id" in df.columns
    else "MAITRI"
)

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