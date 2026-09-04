from fastapi import APIRouter
import pandas as pd

router = APIRouter(
    prefix="/data",
    tags=["Sensor Data"]
)

df = pd.read_csv("Hourly_antarctic_telemetry.csv")


@router.get("/")
def get_all_data():
    data = df.astype(object).where(pd.notnull(df), None)
    return data.to_dict(orient="records")