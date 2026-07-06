from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_name: str = "NeoRAG"
    app_version: str = "1.0.0"
    debug: bool = False
    env: str = "dev"
    secret_key: str = "changeme"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # PostgreSQL (auth user store)
    database_url: str = "postgresql://neorag:neorag@localhost:5432/neorag"

    # Redis (provisioned for future async/background-task use; not yet consumed
    # by application code)
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant (vector search)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # HuggingFace
    huggingface_token: str = ""
    llm_model_name: str = "facebook/opt-1.3b"          # swap for Llama 3.2
    llm_fallback_model: str = "facebook/opt-1.3b"
    llm_max_new_tokens: int = 512                       # tests override this much lower — CPU generation time scales ~linearly with it
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # JWT
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @field_validator("secret_key")
    @classmethod
    def secret_key_not_default_in_prod(cls, v: str, info) -> str:
        # The "changeme" sentinel is only acceptable in dev — applying the
        # length rule to it unconditionally would make the app fail to boot
        # even with its own documented dev default.
        if v == "changeme":
            if info.data.get("env", "dev") != "dev":
                raise ValueError("SECRET_KEY must be set in non-dev environments")
            return v
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v


@lru_cache()
def get_settings() -> Settings:
    return Settings()
