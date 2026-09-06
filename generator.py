import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_telemetry_csv(hours=72, filename="test_antarctic_telemetry.csv"):
    np.random.seed(42)  # Ensures reproducible testing output
    
    start_time = datetime.now() - timedelta(hours=hours)
    data = []
    
    initial_fuel = 20000.0  # Starting liters of diesel
    current_fuel = initial_fuel
    
    for i in range(hours):
        timestamp = start_time + timedelta(hours=i)
        hour = timestamp.hour
        
        # 1. Temperature (°C): Antarctic cold (-38°C to -18°C)
        temp = np.random.uniform(-38.0, -18.0)
        
        # 2. Windspeed (knots)
        windspeed = np.random.uniform(10.0, 50.0)
        
        # 3. Solar Radiation (W/m²): Daylight dependency (peaks mid-day, zero at night)
        if 6 <= hour <= 18:
            solar_radiation = round(max(0, np.sin((hour - 6) * np.pi / 12) * 450 + np.random.normal(0, 15)), 2)
        else:
            solar_radiation = 0.0
            
        # 4. Occupancy (Number of station personnel: 15 to 30)
        occupancy = int(np.random.randint(15, 31))
        
        # 5. Physics logic for Generator Load (%)
        # Lower temp + higher occupancy + lower solar = higher station electrical load
        heating_load = np.abs(temp) * 1.5
        occupancy_load = occupancy * 0.8
        solar_offset = solar_radiation * 0.05
        
        total_load_kw = 60.0 + heating_load + occupancy_load - solar_offset
        generator_load = round(np.clip((total_load_kw / 150.0) * 100, 20.0, 95.0), 2)
        
        # 6. Battery Level (%): Solar charges during peak hours, generator maintains base
        if solar_radiation > 200:
            battery_level = round(np.clip(85.0 + (solar_radiation * 0.03), 85.0, 100.0), 2)
        else:
            battery_level = round(np.clip(100.0 - (generator_load * 0.25), 65.0, 95.0), 2)
            
        # 7. Fuel Level (Liters): Higher generator load burns fuel faster
        burn_rate_lph = generator_load * 0.35  # Liters per hour
        current_fuel -= burn_rate_lph
        fuel_level = round(current_fuel, 2)
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'temp': round(temp, 2),
            'windspeed': round(windspeed, 2),
            'solar_radiation': solar_radiation,
            'occupancy': occupancy,
            'battery_level': battery_level,
            'generator_load': generator_load,
            'fuel_level': fuel_level
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"SUCCESS: Exported {len(df)} rows to '{filename}'")
    print(df.head(5))

if __name__ == "__main__":
    generate_telemetry_csv()