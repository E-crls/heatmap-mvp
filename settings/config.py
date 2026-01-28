"""
Configuration for INPI Heatmap Visual MVP
Environment-based configuration for Docker/Cloud deployment
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()


class Config:
    """Application configuration with environment variable support."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Dataset paths - cache inside repo for Streamlit Cloud
    IMAGES_DIR = Path(os.getenv("IMAGES_DIR", BASE_DIR / "static" / "images"))
    CACHE_DIR = Path(os.getenv("EMBEDDINGS_CACHE_DIR", BASE_DIR / "embeddings_cache"))
    OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", BASE_DIR / "outputs"))
    
    # DINOv3 Configuration
    DINOV3_MODEL_SIZE = os.getenv("DINOV3_MODEL_SIZE", "large")
    
    # Heatmap Configuration
    HEATMAP_THRESHOLD = float(os.getenv("HEATMAP_THRESHOLD", 0.5))
    HEATMAP_ALPHA = float(os.getenv("HEATMAP_ALPHA", 0.6))
    
    # Search Configuration
    TOP_K = int(os.getenv("TOP_K", 5))
    TOP_K_HEATMAPS = int(os.getenv("TOP_K_HEATMAPS", 3))
    
    # Overlay opacities for multiple heatmaps (most similar first)
    OVERLAY_ALPHAS = [0.7, 0.5, 0.3]
    
    # Image extensions supported
    IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (cls.OUTPUTS_DIR / "heatmaps").mkdir(exist_ok=True)
