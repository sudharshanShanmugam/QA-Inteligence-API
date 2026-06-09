from pydantic_settings import BaseSettings
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "qa_intelligence"

    # DeepInfra
    DEEPINFRA_API_KEY: str = ""
    DEEPINFRA_BASE_URL: str = "https://api.deepinfra.com/v1/openai"
    LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    ENTITY_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Graph
    USE_NEO4J: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    GRAPH_PERSIST_PATH: str = "./backend/graph_data/qa_graph.pkl"

    # Vector Store
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "qa_knowledge"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    # Comma-separated allowed CORS origins. Empty = localhost defaults.
    ALLOWED_ORIGINS: str = ""
    # Base directory for all persistent data (projects, analyses, vectors, graph).
    # Set to a mounted disk path in production (e.g. /var/data).
    DATA_DIR: str = ""

    # Processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_RETRIEVAL_DOCS: int = 10

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


settings = Settings()
