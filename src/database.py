from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.config import settings

# If SQLite is used, ensure check_same_thread=False is passed for FastAPI concurrency compatibility
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency injection helper for FastAPI routes to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
