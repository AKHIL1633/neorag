from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "NeoRAG"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "changeme"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # PostgreSQL
    database_url: str = "postgresql://postgres:password@localhost:5432/neorag"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # HuggingFace
    huggingface_token: str = ""
    llm_model_name: str = "facebook/opt-1.3b"          # swap for Llama 3.2
    llm_fallback_model: str = "facebook/opt-1.3b"
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # JWT
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
