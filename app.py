"""
INPI Heatmap Visual MVP - Application Entry Point
Uses DINOv3 (real) + Cross-Similarity for accurate heatmaps
"""
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Use the new API with DINOv3 real + cross-similarity
from api.heatmap_api_v2 import app
from settings.config import Config

if __name__ == "__main__":
    print("="*60)
    print("INPI Heatmap Visual MVP v2.0")
    print("DINOv3 + Cross-Similarity Heatmaps")
    print("="*60)
    print(f"Images directory: {Config.IMAGES_DIR}")
    print(f"Cache directory: {Config.CACHE_DIR}")
    print(f"Server: http://localhost:{Config.API_PORT}")
    print("="*60)
    
    uvicorn.run(
        app, 
        host=Config.API_HOST, 
        port=Config.API_PORT,
        reload=False
    )
