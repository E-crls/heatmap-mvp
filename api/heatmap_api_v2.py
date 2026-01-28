"""
INPI Heatmap Visual MVP - API (DINOv3 Real + Cross-Similarity)
FastAPI backend for visual similarity search with heatmap generation

Uses:
- DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m) for feature extraction
- Cross-similarity technique for accurate heatmaps
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
    from classes.dinov3_extractor_real import DINOv3Extractor
    from classes.heatmap_cross_similarity import CrossSimilarityHeatmap

# Ensure directories exist
Config.ensure_directories()

# Initialize FastAPI app
app = FastAPI(
    title="INPI Heatmap Visual MVP",
    description="Visual similarity search with heatmap visualization using DINOv3",
    version="2.0.0"
)

# CORS middleware
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
    """Get or initialize DINOv3 extractor (real model)."""
    global _extractor
    if _extractor is None:
        from classes.dinov3_extractor_real import DINOv3Extractor
        _extractor = DINOv3Extractor(
            model_size='small',  # Use small for faster inference
            cache_dir=str(Config.CACHE_DIR)
        )
    return _extractor


def get_heatmap_generator():
    """Get or initialize heatmap generator."""
    global _heatmap_gen
    if _heatmap_gen is None:
        from classes.heatmap_cross_similarity import CrossSimilarityHeatmap
        _heatmap_gen = CrossSimilarityHeatmap(alpha=0.5)
    return _heatmap_gen


# Imagens que o usuário pode selecionar como query
SELECTABLE_IMAGES = ["03_chicago_bulls", "input_image", "Audi_logo_detail.svg"]


def get_selectable_images() -> List[Path]:
    """Get list of images that user can SELECT as query (limited set)."""
    result_set = set()  # Use set to avoid duplicates
    for ext in Config.IMAGE_EXTENSIONS:
        for img in Config.IMAGES_DIR.glob(f"*.{ext}"):
            if img.stem in SELECTABLE_IMAGES:
                result_set.add(img)
        for img in Config.IMAGES_DIR.glob(f"*.{ext.upper()}"):
            if img.stem in SELECTABLE_IMAGES:
                result_set.add(img)
    return list(result_set)


def get_dataset_images() -> List[Path]:
    """Get list of ALL images in the dataset (for similarity search)."""
    images_set = set()
    for ext in Config.IMAGE_EXTENSIONS:
        for img in Config.IMAGES_DIR.glob(f"*.{ext}"):
            images_set.add(img)
        for img in Config.IMAGES_DIR.glob(f"*.{ext.upper()}"):
            images_set.add(img)
    return list(images_set)


# Pydantic models
class ImageInfo(BaseModel):
    name: str
    path: str
    url: str


class SimilarImage(BaseModel):
    name: str
    url: str
    similarity: float
    rank: int


class SearchResponse(BaseModel):
    query_image: str
    query_url: str
    similar_images: List[SimilarImage]


class HeatmapResult(BaseModel):
    name: str
    similarity: float
    heatmap_base64: str
    cross_similarity_mean: float
    rank: int


class HeatmapResponse(BaseModel):
    query_image: str
    query_url: str
    heatmaps: List[HeatmapResult]
    overlay_all_base64: Optional[str] = None


# Routes
@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(Config.BASE_DIR / "static" / "index.html")


@app.get("/api/images", response_model=List[ImageInfo])
async def list_images():
    """List only the selectable images (limited set for user selection)."""
    images = get_selectable_images()
    
    return [
        ImageInfo(
            name=img.stem,
            path=str(img),
            url=f"/static/images/{img.name}"
        )
        for img in sorted(images, key=lambda x: x.stem)
    ]


@app.post("/api/search", response_model=SearchResponse)
async def search_similar(image_name: str, top_k: int = 5):
    """
    Find similar images to the selected image.
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
        for img in get_dataset_images():
            if img.stem == image_name or img.name == image_name:
                query_image = img
                break
    
    if query_image is None:
        raise HTTPException(status_code=404, detail=f"Query image not found: {image_name}")
    
    # Get all candidate images (excluding query)
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
    method: str = 'max',
    include_overlay: bool = True
):
    """
    Generate heatmaps for the top-K similar images using cross-similarity.
    
    Args:
        image_name: Name of the query image
        top_k: Number of heatmaps to generate (default: 3)
        method: 'max' or 'topk' for cross-similarity computation
        include_overlay: Whether to include combined overlay of all heatmaps
    """
    extractor = get_dinov3_extractor()
    heatmap_gen = get_heatmap_generator()
    
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
    query_patches, grid_shape = extractor.get_patch_embeddings(query_image)
    
    # Get all candidate images (excluding query)
    all_images = get_dataset_images()
    query_resolved = query_image.resolve()
    candidates = [img for img in all_images if img.resolve() != query_resolved]
    
    # Find similar images
    similar = extractor.find_similar(query_image, candidates, top_k=top_k)
    
    # Generate heatmaps
    heatmaps = []
    target_patches_list = []
    
    for i, (target_path, similarity) in enumerate(similar):
        # Get target patches
        target_patches, _ = extractor.get_patch_embeddings(target_path)
        target_patches_list.append(target_patches)
        
        # Generate heatmap with cross-similarity
        overlay, cross_sim_mean = heatmap_gen.generate_heatmap(
            query_patches, target_patches, query_image, grid_shape,
            method=method, alpha=0.5
        )
        
        # Convert to base64
        heatmap_b64 = heatmap_gen.image_to_base64(overlay)
        
        heatmaps.append(HeatmapResult(
            name=target_path.stem,
            similarity=similarity,
            heatmap_base64=heatmap_b64,
            cross_similarity_mean=cross_sim_mean,
            rank=i + 1
        ))
    
    # Generate combined overlay if requested
    overlay_all_b64 = None
    if include_overlay and target_patches_list:
        overlay_all = heatmap_gen.generate_multiple_heatmaps(
            query_patches, target_patches_list, query_image, grid_shape,
            method=method, alphas=[0.5, 0.35, 0.2]
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
    """Pre-extract and cache embeddings for all dataset images."""
    extractor = get_dinov3_extractor()
    extractor.preload_dataset(Config.IMAGES_DIR)
    return {"status": "success", "message": "All embeddings cached"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m)",
        "heatmap_method": "cross-similarity"
    }
