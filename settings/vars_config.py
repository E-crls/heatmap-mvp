import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    DINOV3_COLLECTION = os.getenv("DINOV3_COLLECTION", "inpi_marcas_dinov3")
    SIGLIP_COLLECTION = os.getenv("SIGLIP_COLLECTION", "inpi_marcas_siglip2")

    TOP_K = int(os.getenv("TOP_K", 10))
    API_PORT = int(os.getenv("API_PORT", 8000))
    HF_TOKEN = os.getenv("HF_TOKEN")
