import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "NexusSearch API"
    DEBUG: bool = False
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./nexus_search.db"
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security Settings
    JWT_SECRET_KEY: str = "super-secret-default-key-change-in-production-1234567890"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # LLM Settings
    LLM_PROVIDER: str = "mock"  # Options: mock, openai, anthropic
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4-turbo"
    
    # Ingestion / indexing path Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    INDEX_DIR: str = "./data_indices"
    FAISS_INDEX_FILENAME: str = "faiss_index.bin"
    BM25_INDEX_FILENAME: str = "bm25_index.pkl"
    DOCUMENTS_PERSIST_FILENAME: str = "documents_store.pkl"
    
    # Prometheus Metrics
    PROMETHEUS_METRICS_PORT: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure indices directory exists
os.makedirs(settings.INDEX_DIR, exist_ok=True)
