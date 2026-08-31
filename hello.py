import time
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# =============================================================================
# 1. DATABASE CONFIGURATION
# =============================================================================
# Change this connection string to match your local MySQL or PostgreSQL setup.
# MySQL: "mysql+pymysql://root:password@localhost:3306/ant_digital_twin"
# Postgres: "postgresql://postgres:password@localhost:5432/ant_digital_twin"
DB_URI = "mysql+pymysql://root:your_mysql_password@localhost:3306/ant_digital_twin"
engine = create_engine(DB_URI)

# Global variables to track state continuity over time
INITIAL_FUEL = 25000.0        # Total starting diesel fuel in Liters
INITIAL_FOOD_RATIONS = 5000.0  # Total starting food supply in Kilograms
CURRENT_FUEL = INITIAL_FUEL
CURRENT_FOOD = INITIAL_FOOD_RATIONS


# =============================================================================
# 2. PHYSICS-INFORMED DATA GENERATOR FUNCTION
# =============================================================================
def generate_minutes_station_reading(station_id="MAITRI"):
    """
    Generates 1 hourly reading containing weather, power, fuel, battery, 
    inventory, and anomaly/disaster labels based on physical relationships.
    """
    global CURRENT_FUEL, CURRENT_FOOD
    
    # --- A. TIME STAMP ---
    current_time = datetime.now()
    hour_of_day = current_time.hour
    
    # --- B. RANDOMIZED SCENARIO TRIGGER (FOR ML ANOMALY & DISASTER INJECTION) ---
    # 80% Normal Operations, 15% Anomaly, 5% Severe Disaster Event
    scenario = np.random.choice(["NORMAL", "ANOMALY", "DISASTER"], p=[0.80, 0.15, 0.05])
    
    # --- C. WEATHER SYNTHESIS ---
    if scenario == "DISASTER":
        # Severe Antarctic Blizzard Event
        temp = float(np.random.uniform(-45.0, -35.0))       # Extreme cold (°C)
        wind = float(np.random.uniform(55.0, 80.0))         # Hurricane-force wind (knots)
        pressure = float(np.random.uniform(950.0, 975.0))   # Low barometric pressure (hPa)
        humidity = float(np.random.uniform(85.0, 98.0))     # High relative humidity (%)
    else:
        # Standard Antarctic Weather Conditions
        temp = float(np.random.uniform(-35.0, -15.0))
        wind = float(np.random.uniform(10.0, 45.0))
        pressure = float(np.random.uniform(985.0, 1015.0))
        humidity = float(np.random.uniform(50.0, 80.0))

    # --- D. STATION PERSONNEL & INVENTORY SYNTHESIS ---
    occupancy = int(np.random.randint(15, 35))  # Active personnel count
    
    # Food ration consumption: Each person eats ~2.0 kg per day (0.083 kg/hr)
    food_consumed_this_hour = occupancy * 0.083
    CURRENT_FOOD = max(0.0, CURRENT_FOOD - food_consumed_this_hour)

    # --- E. ENERGY & POWER SYNTHESIS (PHYSICS LOGIC) ---
    # Rule 1: Colder outdoor temperature directly increases heating power demand
    heating_demand_kw = abs(temp) * 2.1
    base_station_kw = 70.0 + (occupancy * 0.8)
    
    # Rule 2: Calculate total energy load and generator load percentage
    total_power_kw = base_station_kw + heating_demand_kw
    
    if scenario == "ANOMALY":
        # Simulate an unexpected generator mechanical failure / overload spike
        generator_load = float(np.random.uniform(92.0, 99.0))
    else:
        # Standard generator load scaling (200 kW max capacity)
        generator_load = float(np.clip((total_power_kw / 200.0) * 100, 20.0, 90.0))

    # Total energy consumed over 1 hour (kWh = kW * 1 hour)
    energy_consumed_kwh = total_power_kw * 1.0

    # --- F. BATTERY STORAGE SYNTHESIS ---
    # Solar radiation available only during daylight hours (6 AM to 6 PM)
    if 6 <= hour_of_day <= 18 and scenario != "DISASTER":
        solar_kw = float(max(0, np.sin((hour_of_day - 6) * np.pi / 12) * 35.0))
        battery_level = float(np.clip(85.0 + (solar_kw * 0.4), 85.0, 100.0))
    else:
        # Nighttime or blizzard drain on battery backup
        battery_level = float(np.clip(100.0 - (generator_load * 0.3), 15.0, 90.0))

    # --- G. FUEL CONSUMPTION SYNTHESIS ---
    # Rule 3: Generator load percentage dictates diesel burn rate (Liters/hour)
    fuel_burn_rate_lph = generator_load * 0.38
    CURRENT_FUEL = max(0.0, CURRENT_FUEL - fuel_burn_rate_lph)

    # --- H. ANOMALY & DISASTER LABELING (FOR ML TRAINING/PREDICTION) ---
    occurring_anomaly = "NONE"
    predicted_anomaly = "NONE"
    disaster_label = "NONE"
    
    # Evaluate ML Labels based on metrics
    if scenario == "DISASTER":
        disaster_label = "DISASTER_BLIZZARD_EMERGENCY"
    elif CURRENT_FUEL < 2000.0:
        disaster_label = "DISASTER_FUEL_EXHAUSTION_CRITICAL"
        
    if scenario == "ANOMALY" or generator_load > 90.0:
        occurring_anomaly = "ANOMALY_GENERATOR_OVERLOAD"
    elif battery_level < 25.0:
        occurring_anomaly = "ANOMALY_BATTERY_CRITICAL_DEPLETION"
        
    # Predictive warning flags (Trend analysis for AI model evaluation)
    hours_of_fuel_remaining = CURRENT_FUEL / max(fuel_burn_rate_lph, 0.1)
    if hours_of_fuel_remaining < 48.0:
        predicted_anomaly = "PREDICTED_FUEL_DEPLETION_48H"
    elif temp < -30.0 and battery_level < 40.0:
        predicted_anomaly = "PREDICTED_HEATING_FAILURE_RISK"

    # --- I. ASSEMBLE DATAFRAME ROW ---
    record = {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'station_id': station_id,
        
        # Weather Fields
        'temperature_celsius': round(temp, 2),
        'wind_speed_knots': round(wind, 2),
        'pressure_hpa': round(pressure, 2),
        'humidity_percent': round(humidity, 2),
        
        # Power & Energy Fields
        'generator_load_percent': round(generator_load, 2),
        'energy_consumed_kwh': round(energy_consumed_kwh, 2),
        'battery_level_percent': round(battery_level, 2),
        
        # Fuel & Inventory Fields
        'fuel_level_liters': round(CURRENT_FUEL, 2),
        'fuel_burn_rate_lph': round(fuel_burn_rate_lph, 2),
        'food_inventory_kg': round(CURRENT_FOOD, 2),
        'station_occupancy': occupancy,
        
        # Machine Learning Target Labels
        'occurring_anomaly': occurring_anomaly,
        'predicted_anomaly': predicted_anomaly,
        'disaster_label': disaster_label
    }
    
    return pd.DataFrame([record])


# =============================================================================
# 3. REAL-TIME EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" HOURLY REAL-TIME ANTARCTIC TELEMETRY GENERATOR STARTED")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # SET INTERVAL MODE:
    # Set to 3600 for true 1-hour real-time intervals.
    # Set to 5 for rapid demo/testing mode during judge presentation.
    # -------------------------------------------------------------------------
    INTERVAL_SECONDS = 10 
    
    print(f"Data Generation Interval: Every {INTERVAL_SECONDS} Seconds")
    print("Press Ctrl + C to stop the generator script.\n")
    
    try:
        while True:
            # 1. Generate 1 hour of telemetry data
            telemetry_df = generate_minutes_station_reading(station_id="MAITRI")
            
            # 2. Print formatting to VS Code console
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] New Telemetry Generated:")
            print(telemetry_df.to_string(index=False))
            print("-" * 80)
            
            # 3. Export/Backup locally to CSV
            telemetry_df.to_csv("hourly_antarctic_telemetry.csv", mode='a', header=not pd.io.common.file_exists("hourly_antarctic_telemetry.csv"), index=False)
            
            # 4. Push directly to SQL Database (PostgreSQL/MySQL)
            try:
                telemetry_df.to_sql('hourly_telemetry', engine, if_exists='append', index=False)
                print(" -> Status: Written to SQL Database successfully.")
            except Exception as e:
                print(f" -> DB Notice: Saved to CSV (SQL write skipped: {e})")
                
            print(f"Waiting {INTERVAL_SECONDS} seconds for the next reading...\n")
            
            # 5. Sleep until next 1-hour schedule
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nGenerator script gracefully stopped by user.")