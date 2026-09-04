import time
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# =============================================================================
# 1. DATABASE CONFIGURATION
# =============================================================================
# MySQL: "mysql+pymysql://root:password@localhost:3306/ant_digital_twin"
# Postgres: "postgresql://postgres:password@localhost:5432/ant_digital_twin"
DB_URI = "mysql+pymysql://root:your_mysql_password@localhost:3306/ant_digital_twin"
engine = create_engine(DB_URI)

# Global state trackers for resource depletion continuity
INITIAL_FUEL = 25000.0        # Total starting diesel fuel in Liters
INITIAL_FOOD_RATIONS = 5000.0  # Total starting food supply in Kilograms
CURRENT_FUEL = INITIAL_FUEL
CURRENT_FOOD = INITIAL_FOOD_RATIONS

# Interval configuration (seconds between ticks)
# Use 10 for demo streaming; use 3600 for true 1-hour real-time intervals.
INTERVAL_SECONDS = 10  


# =============================================================================
# 2. PHYSICS-INFORMED DATA GENERATOR FUNCTION
# =============================================================================
def generate_station_reading(station_id="MAITRI"):
    """
    Generates 1 telemetry reading containing weather, power, fuel, battery, 
    inventory, and dynamic ML anomaly/disaster target labels.
    """
    global CURRENT_FUEL, CURRENT_FOOD
    
    # --- A. TIME STAMP ---
    current_time = datetime.now()
    hour_of_day = current_time.hour
    
    # --- B. RANDOMIZED SCENARIO TRIGGER (FOR ML ANOMALY & DISASTER INJECTION) ---
    # 80% Normal Operations, 15% Anomaly Event, 5% Severe Disaster Event
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
    
    # Scale depletion based on iteration interval length
    time_fraction_of_hour = INTERVAL_SECONDS / 3600.0
    
    # Food consumption (~2.0 kg/person/day = 0.083 kg/person/hour)
    food_consumed = (occupancy * 0.083) * time_fraction_of_hour
    CURRENT_FOOD = max(0.0, CURRENT_FOOD - food_consumed)

    # --- E. ENERGY & POWER SYNTHESIS (PHYSICS LOGIC) ---
    # Rule 1: Colder outdoor temperature directly increases heating power demand
    heating_demand_kw = abs(temp) * 2.1
    base_station_kw = 70.0 + (occupancy * 0.8)
    
    # Rule 2: Calculate total energy load and generator load percentage
    total_power_kw = base_station_kw + heating_demand_kw
    
    if scenario == "ANOMALY":
        # Simulate unexpected generator overload/mechanical stress
        generator_load = float(np.random.uniform(92.0, 99.0))
    else:
        # Standard generator load scaling (200 kW max capacity)
        generator_load = float(np.clip((total_power_kw / 200.0) * 100, 20.0, 90.0))

    # Total energy consumed over the elapsed interval step
    energy_consumed_kwh = total_power_kw * time_fraction_of_hour

    # --- F. BATTERY STORAGE SYNTHESIS ---
    # Solar radiation available during daytime (6 AM to 6 PM)
    if 6 <= hour_of_day <= 18 and scenario != "DISASTER":
        solar_kw = float(max(0, np.sin((hour_of_day - 6) * np.pi / 12) * 35.0))
        battery_level = float(np.clip(85.0 + (solar_kw * 0.4), 85.0, 100.0))
    else:
        # Nighttime or blizzard drain on battery backup
        solar_kw=0.0
        battery_level = float(np.clip(100.0 - (generator_load * 0.3), 15.0, 90.0))

    # --- G. FUEL CONSUMPTION SYNTHESIS ---
    # Rule 3: Generator load dictates hourly burn rate (Liters/hour)
    fuel_burn_rate_lph = generator_load * 0.38
    fuel_used_this_interval = fuel_burn_rate_lph * time_fraction_of_hour
    CURRENT_FUEL = max(0.0, CURRENT_FUEL - fuel_used_this_interval)

    # --- H. ANOMALY & DISASTER LABELING (FOR ML EVALUATION) ---
    # occurring_anomaly = "NONE"
    # predicted_anomaly = "NONE"
    # disaster_label = "NONE"
    
    # # Evaluate Disaster Conditions
    # if scenario == "DISASTER":
    #     disaster_label = "DISASTER_BLIZZARD_EMERGENCY"
    # elif CURRENT_FUEL < 2000.0:
    #     disaster_label = "DISASTER_FUEL_EXHAUSTION_CRITICAL"
        
    # Evaluate Occurring Anomalies
    # if scenario == "ANOMALY" or generator_load > 90.0:
    #     occurring_anomaly = "ANOMALY_GENERATOR_OVERLOAD"
    # elif battery_level < 25.0:
    #     occurring_anomaly = "ANOMALY_BATTERY_CRITICAL_DEPLETION"
        
    # # Evaluate Predictive Risk Trends
    # hours_of_fuel_remaining = CURRENT_FUEL / max(fuel_burn_rate_lph, 0.1)
    # if hours_of_fuel_remaining < 48.0:
    #     predicted_anomaly = "PREDICTED_FUEL_DEPLETION_48H"
    # elif temp < -30.0 and battery_level < 40.0:
    #     predicted_anomaly = "PREDICTED_HEATING_FAILURE_RISK"

    # --- I. ASSEMBLE DATAFRAME ROW ---
    record = {
        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
        'station_id': station_id,
        
        # Weather Fields
        'Temperature': round(temp, 2),
        'Wind_Speed': round(wind, 2),
        'Pressure': round(pressure, 2),
        'Humidity': round(humidity, 2),
        
        # Power & Energy Fields
        'Solar_Radiation':round(solar_kw,2),
        'Generator_Load': round(generator_load, 2),
        'Energy_consumption': round(energy_consumed_kwh, 2),
        'Battery_Level': round(battery_level, 2),
        
        # Fuel & Inventory Fields
        'Fuel_Level': round(CURRENT_FUEL, 2),
        'Fuel_Burn_Rate': round(fuel_burn_rate_lph, 2),
        'Food_Inventory': round(CURRENT_FOOD, 2),
        'Occupancy': occupancy,
        
        # Machine Learning Target Labels
        # 'occurring_anomaly': occurring_anomaly,
        # 'predicted_anomaly': predicted_anomaly,
        # 'disaster_label': disaster_label
    }
    
    return pd.DataFrame([record])


# =============================================================================
# 3. REAL-TIME EXECUTION LOOP
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" ANTARCTIC TELEMETRY GENERATOR RUNNING")
    print("=" * 80)
    print(f"Data Generation Interval: Every {INTERVAL_SECONDS} Seconds")
    print("Press Ctrl + C to stop the script.\n")
    
    try:
        while True:
            # 1. Generate telemetry reading
            telemetry_df = generate_station_reading(station_id="MAITRI")
            
            # 2. Output to console
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Telemetry Generated:")
            print(telemetry_df.to_string(index=False))
            print("-" * 80)
            
            # 3. Export to local CSV backup
            telemetry_df.to_csv("Hourly_antarctic_telemetry.csv", mode='a', header=not pd.io.common.file_exists("hourly_antarctic_telemetry.csv"), index=False)
            
            # 4. Push to database
            try:
                telemetry_df.to_sql('hourly_telemetry', engine, if_exists='append', index=False)
                print(" -> Status: Written to SQL Database successfully.")
            except Exception as e:
                print(f" -> DB Notice: Saved to CSV (SQL write skipped: {e})")
                
            print(f"Waiting {INTERVAL_SECONDS} seconds for the next reading...\n")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nGenerator script stopped.")