from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# -------------------------------------------------------------
# Database Configuration (Fill these in once you have the info)
# -------------------------------------------------------------
DB_USER = "root"
DB_PASSWORD = "12345"
DB_HOST = "localhost"        # or your server's IP address
DB_PORT = "3306"             # 3306 for MySQL, 5432 for PostgreSQL
DB_NAME = "BHOSDI"

# Example connection string for MySQL using PyMySQL:
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI App Setup
app = FastAPI(title="Telemetry API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Set specific origins (e.g., ["http://localhost:3000"]) in production
    allow_credentials=True,
    allow_methods=["*"],          # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)


# Database Session Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# API Endpoints


# 1. Fetch the single latest reading (best for live real-time dashboard cards)
@app.get("/api/telemetry/latest")
def get_latest_reading(db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM hourly_telemetry ORDER BY timestamp DESC LIMIT 1")
        result = db.execute(query).mappings().first()
        if not result:
            raise HTTPException(status_code=404, detail="No telemetry records found.")
        return dict(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Fetch recent history (useful for frontend charts/graphs)
@app.get("/api/telemetry/history")
def get_telemetry_history(limit: int = 50, db: Session = Depends(get_db)):
    try:
        query = text("SELECT * FROM hourly_telemetry ORDER BY timestamp DESC LIMIT :limit")
        result = db.execute(query, {"limit": limit}).mappings().all()
        return [dict(row) for row in result]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))