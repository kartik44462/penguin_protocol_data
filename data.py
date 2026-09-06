from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DataInput(BaseModel):

    timestamp: datetime

    station_id: str

    temperature_celsius: float = Field(
        ...,
        description="Temperature in Celsius"
    )

    wind_speed_knots: float = Field(
        ...,
        ge=0,
        description="Wind speed in knots"
    )

    pressure_hpa: float = Field(
        ...,
        ge=0,
        description="Atmospheric pressure in hPa"
    )

    humidity_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Humidity percentage"
    )

    generator_load_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Generator load percentage"
    )

    energy_consumed_kwh: float = Field(
        ...,
        ge=0,
        description="Energy consumed in kWh"
    )

    battery_level_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Battery level percentage"
    )

    fuel_level_liters: float = Field(
        ...,
        ge=0,
        description="Fuel level in liters"
    )

    fuel_burn_rate_lph: float = Field(
        ...,
        ge=0,
        description="Fuel burn rate in liters per hour"
    )

    food_inventory_kg: float = Field(
        ...,
        ge=0,
        description="Food inventory in kilograms"
    )

    station_occupancy: int = Field(
        ...,
        ge=0,
        description="Number of people at the station"
    )

    occurring_anomaly: Optional[str] = None

    predicted_anomaly: Optional[str] = None

    disaster_label: Optional[str] = None


class DataResponse(BaseModel):

    message: str

    data: DataInput