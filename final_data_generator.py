import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. SETTINGS & FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REAL_DATA_FILE = BASE_DIR / "Maitri - AWS_2016_filtered_data.xlsx"
CSV_FILE = BASE_DIR / "realtime_antarctic_data.csv"

STATION_ID = "MAITRI"

# Stream 1 row every 10 real-world seconds
INTERVAL_SECONDS = 10

# Global state trackers
fuel = 25000.0  # Liters
food = 5000.0  # Kilograms
battery = 75.0  # Percentage (%)
occupancy = 25  # Active personnel

# Dataset iterator index tracker
weather_data_index = 0


# ============================================================
# 2. LOAD REAL MAITRI AWS DATA
# ============================================================

print("Loading real Maitri AWS data...")

if not REAL_DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the dataset at: {REAL_DATA_FILE}. "
        "Please place the Maitri Excel file in the script directory."
    )

real_data = pd.read_excel(REAL_DATA_FILE, engine="openpyxl")

# Clean timestamp and numeric columns
real_data["obstime"] = pd.to_datetime(real_data["obstime"], errors="coerce")

for column in ["tempr", "ap", "ws", "rh"]:
    real_data[column] = pd.to_numeric(real_data[column], errors="coerce")

real_data = real_data.dropna(
    subset=["obstime", "tempr", "ap", "ws", "rh"]
).reset_index(drop=True)

total_rows = len(real_data)
print(f"Loaded {total_rows:,} sequential Maitri AWS weather readings.\n")


# ============================================================
# 3. FUNCTION TO GET SEQUENTIAL WEATHER
# ============================================================


def get_sequential_weather():
    """Reads weather sequentially from the dataset to preserve smooth weather trends."""
    global weather_data_index

    row = real_data.iloc[weather_data_index]

    # Increment index and loop back to start if reaching dataset end
    weather_data_index = (weather_data_index + 1) % total_rows

    temperature = float(row["tempr"])
    wind_speed = float(row["ws"])
    pressure = float(row["ap"])
    humidity = float(row["rh"])

    return temperature, wind_speed, pressure, humidity


# ============================================================
# 4. GENERATE ONE TELEMETRY ROW
# ============================================================


def generate_one_row():
    global fuel, food, battery, occupancy

    # --- REAL-TIME SYSTEM TIMESTAMP ---
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    hour_of_day = now.hour

    # Fetch continuous sequential weather reading
    temperature, wind_speed, pressure, humidity = get_sequential_weather()

    # Dynamic occupancy fluctuation
    occupancy += int(np.random.choice([-1, 0, 0, 0, 1]))
    occupancy = int(np.clip(occupancy, 15, 35))

    # --- SOLAR RADIATION PHYSICS ---
    # Based on current actual hour of day
    if 6 <= hour_of_day <= 18:
        daylight_progress = (hour_of_day - 6) / 12.0
        clear_sky_peak = np.sin(daylight_progress * np.pi) * 800.0
        cloud_factor = max(0.2, 1.0 - (humidity / 150.0))
        solar_radiation = clear_sky_peak * cloud_factor
    else:
        solar_radiation = 0.0

    solar_radiation = max(0.0, float(solar_radiation))

    # --- ENERGY & HEATING DEMAND ---
    heating = max(0.0, abs(temperature) - 5.0) * 1.2
    base_power = 60.0 + (occupancy * 0.8)
    energy_consumption = base_power + heating

    # --- GENERATOR LOAD ---
    generator_load = (energy_consumption / 200.0) * 100.0
    generator_load = float(np.clip(generator_load, 20.0, 90.0))

    # --- TIME SCALE FACTOR ---
    # Scaled down for a 10-second tick interval
    time_scale = INTERVAL_SECONDS / 3600.0

    # --- FUEL CONSUMPTION ---
    fuel_burn_rate = generator_load * 0.38  # Liters per hour
    fuel_used_this_tick = fuel_burn_rate * time_scale
    fuel = max(0.0, fuel - fuel_used_this_tick)

    # --- BATTERY STORAGE BALANCING ---
    solar_charge_rate = (solar_radiation / 800.0) * 2.5  # Max +2.5% per hour
    generator_drain_rate = (generator_load / 100.0) * 1.0  # Max -1.0% per hour
    net_battery_change = (
        solar_charge_rate - generator_drain_rate
    ) * time_scale

    battery = float(np.clip(battery + net_battery_change, 0.0, 100.0))

    # --- FOOD CONSUMPTION ---
    food_consumed_hourly = occupancy * (2.0 / 24.0)  # ~2kg/person/day
    food_used_this_tick = food_consumed_hourly * time_scale
    food = max(0.0, food - food_used_this_tick)

    # --- ASSEMBLE TELEMETRY ROW ---
    row = {
        "timestamp": timestamp_str,
        "station_id": STATION_ID,
        "Temperature": round(temperature, 2),
        "Wind_Speed": round(wind_speed, 2),
        "Pressure": round(pressure, 2),
        "Humidity": round(humidity, 2),
        "Solar_Radiation": round(solar_radiation, 2),
        "Generator_Load": round(generator_load, 2),
        "Energy_consumption": round(energy_consumption, 2),
        "Battery_Level": round(battery, 2),
        "Fuel_Level": round(fuel, 2),
        "Fuel_Burn_Rate": round(fuel_burn_rate, 2),
        "Food_Inventory": round(food, 2),
        "Station_Occupancy": occupancy,
    }

    return row


# ============================================================
# 5. REAL-TIME LOOP
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" MAITRI REAL-TIME TELEMETRY GENERATOR")
    print("=" * 70)
    print(f"Station ID: {STATION_ID}")
    print(f"Streaming frequency: 1 row every {INTERVAL_SECONDS} seconds")
    print("Press Ctrl + C to stop script.\n")
    print("=" * 70 + "\n")

    try:
        while True:
            # Generate row
            row_data = generate_one_row()
            df_row = pd.DataFrame([row_data])

            # Output to console
            print(df_row.to_string(index=False))

            # Append to local CSV file
            df_row.to_csv(
                CSV_FILE,
                mode="a",
                header=not CSV_FILE.exists(),
                index=False,
            )

            print(f"\n[Saved to CSV] {CSV_FILE}")
            print("-" * 70)

            # Sleep interval
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nGenerator stopped by user.")