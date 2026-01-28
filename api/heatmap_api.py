"""
INPI Heatmap Visual MVP - API
FastAPI backend for visual similarity search with heatmap generation
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
import os
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings.config import Config

# Lazy imports for heavy ML modules
if TYPE_CHECKING:
    from classes.dinov3_extractor import DINOv3Extractor
    from classes.heatmap_generator import HeatmapGenerator

# Ensure directories exist
Config.ensure_directories()

# Initialize FastAPI app
app = FastAPI(
    title="INPI Heatmap Visual MVP",
    description="Visual similarity search with heatmap visualization for INPI brand comparison",
    version="1.0.0"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Config.BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global instances (lazy loaded)
_extractor = None
_heatmap_gen = None


def get_dinov3_extractor():
    """Get or initialize DINOv3 extractor."""
    global _extractor
    if _extractor is None:
        from classes.dinov3_extractor import DINOv3Extractor
        _extractor = DINOv3Extractor(
            model_size=Config.DINOV3_MODEL_SIZE,
            cache_dir=str(Config.CACHE_DIR)
        )
    return _extractor


def get_generator():
    """Get or initialize heatmap generator."""
    global _heatmap_gen
    if _heatmap_gen is None:
        from classes.heatmap_generator import HeatmapGenerator
        _heatmap_gen = HeatmapGenerator(
            threshold=Config.HEATMAP_THRESHOLD,
            alpha=Config.HEATMAP_ALPHA
        )
    return _heatmap_gen


def get_dataset_images() -> List[Path]:
    """Get list of all images in the dataset (no duplicates)."""
    images_set = set()
    for ext in Config.IMAGE_EXTENSIONS:
        # Use case-insensitive matching by converting to lowercase for comparison
        for img in Config.IMAGES_DIR.glob(f"*.{ext}"):
            images_set.add(img)
        for img in Config.IMAGES_DIR.glob(f"*.{ext.upper()}"):
            images_set.add(img)
    # Convert to list and sort
    return sorted(list(images_set), key=lambda x: x.name.lower())


# ============================================================================
# Pydantic Models
# ============================================================================

class ImageInfo(BaseModel):
    """Information about a dataset image."""
    name: str
    path: str
    url: str


class SimilarImage(BaseModel):
    """Similar image result."""
    name: str
    url: str
    similarity: float
    rank: int


class HeatmapResult(BaseModel):
    """Heatmap generation result."""
    name: str
    similarity: float
    heatmap_base64: str
    rank: int


class SearchResponse(BaseModel):
    """Response for similarity search."""
    query_image: str
    query_url: str
    similar_images: List[SimilarImage]


class HeatmapResponse(BaseModel):
    """Response for heatmap generation."""
    query_image: str
    query_url: str
    heatmaps: List[HeatmapResult]
    overlay_all_base64: Optional[str] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(static_path / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "images_count": len(get_dataset_images()),
        "cache_dir": str(Config.CACHE_DIR),
        "model_loaded": _extractor is not None
    }


@app.get("/api/images", response_model=List[ImageInfo])
async def list_images():
    """List all available images in the dataset."""
    images = get_dataset_images()
    
    return [
        ImageInfo(
            name=img.stem,
            path=str(img),
            url=f"/static/images/{img.name}"
        )
        for img in images
    ]


@app.get("/api/image/{image_name}")
async def get_image(image_name: str):
    """Get a specific image by name."""
    images_dir = Config.IMAGES_DIR
    
    # Try different extensions
    for ext in Config.IMAGE_EXTENSIONS:
        path = images_dir / f"{image_name}.{ext}"
        if path.exists():
            return FileResponse(path)
        path = images_dir / f"{image_name}.{ext.upper()}"
        if path.exists():
            return FileResponse(path)
    
    # Try exact match with extension
    path = images_dir / image_name
    if path.exists():
        return FileResponse(path)
    
    raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")


@app.post("/api/search", response_model=SearchResponse)
async def search_similar(image_name: str, top_k: int = 5):
    """
    Find similar images to the selected image.
    
    Args:
        image_name: Name of the query image (without extension)
        top_k: Number of similar images to return
    """
    extractor = get_dinov3_extractor()
    
    # Find query image
    query_image = None
    for ext in Config.IMAGE_EXTENSIONS:
        path = Config.IMAGES_DIR / f"{image_name}.{ext}"
        if path.exists():
            query_image = path
            break
    
    if query_image is None:
        # Try with exact name
        for img in get_dataset_images():
            if img.stem == image_name or img.name == image_name:
                query_image = img
                break
    
    if query_image is None:
        raise HTTPException(status_code=404, detail=f"Query image not found: {image_name}")
    
    # Get all candidate images (excluding query by comparing resolved paths)
    all_images = get_dataset_images()
    query_resolved = query_image.resolve()
    candidates = [img for img in all_images if img.resolve() != query_resolved]
    
    if not candidates:
        raise HTTPException(status_code=400, detail="No other images available for comparison")
    
    # Find similar images
    similar = extractor.find_similar(query_image, candidates, top_k=top_k)
    
    return SearchResponse(
        query_image=query_image.stem,
        query_url=f"/static/images/{query_image.name}",
        similar_images=[
            SimilarImage(
                name=img_path.stem,
                url=f"/static/images/{img_path.name}",
                similarity=score,
                rank=i + 1
            )
            for i, (img_path, score) in enumerate(similar)
        ]
    )


@app.post("/api/heatmap", response_model=HeatmapResponse)
async def generate_heatmaps(
    image_name: str, 
    top_k: int = 3,
    threshold: Optional[float] = None,
    include_overlay: bool = True
):
    """
    Generate heatmaps for the top-K similar images.
    
    Args:
        image_name: Name of the query image
        top_k: Number of heatmaps to generate (default: 3)
        threshold: Intensity threshold (0-1). Pixels below become transparent.
        include_overlay: Whether to include combined overlay of all heatmaps
    """
    extractor = get_dinov3_extractor()
    heatmap_gen = get_generator()
    
    if threshold is not None:
        heatmap_gen.threshold = threshold
    
    # Find query image
    query_image = None
    for ext in Config.IMAGE_EXTENSIONS:
        path = Config.IMAGES_DIR / f"{image_name}.{ext}"
        if path.exists():
            query_image = path
            break
    
    if query_image is None:
        for img in get_dataset_images():
            if img.stem == image_name or img.name == image_name:
                query_image = img
                break
    
    if query_image is None:
        raise HTTPException(status_code=404, detail=f"Query image not found: {image_name}")
    
    # Get query embeddings
    query_patches = extractor.get_patch_embeddings(query_image)
    
    # Get all candidate images (excluding query by comparing resolved paths)
    all_images = get_dataset_images()
    query_resolved = query_image.resolve()
    candidates = [img for img in all_images if img.resolve() != query_resolved]
    
    # Find similar images
    similar = extractor.find_similar(query_image, candidates, top_k=top_k)
    
    # Generate heatmaps
    heatmaps = []
    similarity_grids = []
    
    for i, (target_path, similarity) in enumerate(similar):
        # Get target patches
        target_patches = extractor.get_patch_embeddings(target_path)
        
        # Generate heatmap
        overlay, _ = heatmap_gen.generate_heatmap(
            query_patches,
            target_patches,
            query_image,
            alpha=Config.OVERLAY_ALPHAS[i] if i < len(Config.OVERLAY_ALPHAS) else 0.5
        )
        
        # Store grid for combined overlay
        sim_vector = heatmap_gen.compute_patch_similarity(query_patches, target_patches)
        sim_grid = heatmap_gen.reshape_to_grid(sim_vector)
        similarity_grids.append(sim_grid)
        
        # Convert to base64
        heatmap_b64 = heatmap_gen.image_to_base64(overlay)
        
        heatmaps.append(HeatmapResult(
            name=target_path.stem,
            similarity=similarity,
            heatmap_base64=heatmap_b64,
            rank=i + 1
        ))
    
    # Generate combined overlay if requested
    overlay_all_b64 = None
    if include_overlay and similarity_grids:
        overlay_all = heatmap_gen.overlay_multiple(
            query_image,
            similarity_grids,
            alphas=Config.OVERLAY_ALPHAS[:len(similarity_grids)]
        )
        overlay_all_b64 = heatmap_gen.image_to_base64(overlay_all)
    
    return HeatmapResponse(
        query_image=query_image.stem,
        query_url=f"/static/images/{query_image.name}",
        heatmaps=heatmaps,
        overlay_all_base64=overlay_all_b64
    )


@app.post("/api/preload")
async def preload_embeddings():
    """
    Pre-extract and cache embeddings for all dataset images.
    Call this once to warm up the cache.
    """
    extractor = get_dinov3_extractor()
    
    images = get_dataset_images()
    results = []
    
    for img in images:
        cached = extractor.is_cached(img)
        if not cached:
            extractor.extract_features(img, use_cache=True)
        
        results.append({
            "name": img.stem,
            "was_cached": cached,
            "status": "ok"
        })
    
    return {
        "status": "success",
        "total_images": len(images),
        "results": results
    }


@app.get("/api/cache/status")
async def cache_status():
    """Check cache status for all dataset images."""
    extractor = get_dinov3_extractor()
    images = get_dataset_images()
    
    cached = []
    not_cached = []
    
    for img in images:
        if extractor.is_cached(img):
            cached.append(img.stem)
        else:
            not_cached.append(img.stem)
    
    return {
        "total_images": len(images),
        "cached_count": len(cached),
        "not_cached_count": len(not_cached),
        "cached": cached,
        "not_cached": not_cached,
        "cache_dir": str(Config.CACHE_DIR)
    }
