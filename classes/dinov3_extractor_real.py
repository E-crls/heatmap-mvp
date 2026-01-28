"""
DINOv3 Feature Extractor - REAL DINOv3 from Hugging Face
Model: facebook/dinov3-vits16-pretrain-lvd1689m (or larger variants)

This replaces the old DINOv2-based extractor with the actual DINOv3 model.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional, List
import os

from transformers import AutoImageProcessor, AutoModel


class DINOv3Extractor:
    """
    DINOv3 feature extractor using Hugging Face Transformers.
    Extracts CLS token for similarity search and patch embeddings for heatmaps.
    """
    
    AVAILABLE_MODELS = {
        'small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'base': 'facebook/dinov3-vitb16-pretrain-lvd1689m', 
        'large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    }
    
    def __init__(self, model_size: str = 'small', cache_dir: Optional[str] = None):
        """
        Initialize the DINOv3 extractor.
        
        Args:
            model_size: Model size ('small', 'base', 'large')
            cache_dir: Directory to store embedding cache.
        """
        self.model_size = model_size
        self.model_name = self.AVAILABLE_MODELS.get(model_size, self.AVAILABLE_MODELS['small'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            base_dir = Path(__file__).parent.parent.parent
            self.cache_dir = Path(os.getenv('EMBEDDINGS_CACHE_DIR', base_dir / 'embeddings_cache_v3'))
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = None
        self.processor = None
        self.patch_size = None
        self.num_register_tokens = None
        self._load_model()
    
    def _load_model(self):
        """Load DINOv3 model from Hugging Face"""
        print(f"[DINOv3] Loading {self.model_name} on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        
        print(f"[DINOv3] Model loaded. Patch size: {self.patch_size}, Register tokens: {self.num_register_tokens}")
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate a cache key from image path"""
        path = Path(image_path)
        return f"{path.stem}_dinov3_real.npz"
    
    def _get_cache_path(self, image_path: Union[str, Path]) -> Path:
        """Get full cache file path for an image"""
        return self.cache_dir / self._get_cache_key(image_path)
    
    def is_cached(self, image_path: Union[str, Path]) -> bool:
        """Check if embeddings are cached for an image"""
        return self._get_cache_path(image_path).exists()
    
    def extract_features(self, image: Union[str, Path, Image.Image], 
                        use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Extract DINOv3 features from an image.
        
        Args:
            image: Image path or PIL Image
            use_cache: Whether to use/save cache (only works with paths)
        
        Returns:
            Tuple of (cls_embedding, patch_embeddings, grid_shape)
            - cls_embedding: [hidden_size] - for similarity search
            - patch_embeddings: [num_patches, hidden_size] - for heatmaps
            - grid_shape: (grid_h, grid_w) - spatial dimensions of patches
        """
        image_path = None
        
        # If image is a path, check cache
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            
            if use_cache and self.is_cached(image_path):
                return self._load_from_cache(image_path)
            
            # Load image from path
            pil_image = Image.open(image_path).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state  # [1, 1 + num_register + num_patches, hidden_size]
        
        # Extract CLS token
        cls_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        # Extract patch features (skip CLS and register tokens)
        patch_embeddings = last_hidden_state[:, 1 + self.num_register_tokens:, :].cpu().numpy()
        patch_embeddings = patch_embeddings[0]  # Remove batch dim: [num_patches, hidden_size]
        
        # Calculate grid shape
        _, _, img_height, img_width = inputs.pixel_values.shape
        grid_h = img_height // self.patch_size
        grid_w = img_width // self.patch_size
        grid_shape = (grid_h, grid_w)
        
        # Save to cache if we have a path
        if image_path and use_cache:
            self._save_to_cache(image_path, cls_embedding, patch_embeddings, grid_shape)
        
        return cls_embedding, patch_embeddings, grid_shape
    
    def _load_from_cache(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Load embeddings from cache"""
        cache_path = self._get_cache_path(image_path)
        data = np.load(cache_path)
        grid_shape = tuple(data['grid_shape'])
        return data['cls_embedding'], data['patch_embeddings'], grid_shape
    
    def _save_to_cache(self, image_path: Union[str, Path], 
                       cls_embedding: np.ndarray, 
                       patch_embeddings: np.ndarray,
                       grid_shape: Tuple[int, int]):
        """Save embeddings to cache"""
        cache_path = self._get_cache_path(image_path)
        np.savez_compressed(
            cache_path,
            cls_embedding=cls_embedding,
            patch_embeddings=patch_embeddings,
            grid_shape=np.array(grid_shape),
            image_path=str(image_path)
        )
    
    def get_cls_embedding(self, image: Union[str, Path, Image.Image], 
                          use_cache: bool = True) -> np.ndarray:
        """
        Get only the CLS embedding (for similarity search).
        
        Returns:
            cls_embedding: [hidden_size] - flattened for vector search
        """
        cls_emb, _, _ = self.extract_features(image, use_cache)
        return cls_emb
    
    def get_patch_embeddings(self, image: Union[str, Path, Image.Image],
                             use_cache: bool = True) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Get patch embeddings and grid shape (for heatmap generation).
        
        Returns:
            Tuple of (patch_embeddings, grid_shape)
            - patch_embeddings: [num_patches, hidden_size]
            - grid_shape: (grid_h, grid_w)
        """
        _, patch_emb, grid_shape = self.extract_features(image, use_cache)
        return patch_emb, grid_shape
    
    def compute_similarity(self, image1: Union[str, Path, Image.Image],
                          image2: Union[str, Path, Image.Image]) -> float:
        """
        Compute cosine similarity between two images using CLS tokens.
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        emb1 = self.get_cls_embedding(image1)
        emb2 = self.get_cls_embedding(image2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def find_similar(self, query_image: Union[str, Path, Image.Image],
                    candidate_images: List[Union[str, Path]],
                    top_k: int = 5) -> List[Tuple[Path, float]]:
        """
        Find most similar images from a list of candidates.
        
        Args:
            query_image: Query image
            candidate_images: List of candidate image paths
            top_k: Number of results to return
        
        Returns:
            List of (image_path, similarity_score) tuples, sorted by similarity
        """
        query_emb = self.get_cls_embedding(query_image)
        
        results = []
        for img_path in candidate_images:
            path = Path(img_path)
            if path.exists():
                candidate_emb = self.get_cls_embedding(path)
                similarity = np.dot(query_emb, candidate_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(candidate_emb)
                )
                results.append((path, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def preload_dataset(self, image_dir: Union[str, Path], 
                        extensions: List[str] = ['png', 'jpg', 'jpeg', 'webp']):
        """
        Pre-extract and cache embeddings for all images in a directory.
        """
        image_dir = Path(image_dir)
        
        all_images = []
        for ext in extensions:
            all_images.extend(image_dir.glob(f"*.{ext}"))
            all_images.extend(image_dir.glob(f"*.{ext.upper()}"))
        
        print(f"[DINOv3] Pre-loading {len(all_images)} images from {image_dir}")
        
        for i, img_path in enumerate(all_images, 1):
            if self.is_cached(img_path):
                print(f"  [{i}/{len(all_images)}] CACHED: {img_path.name}")
            else:
                print(f"  [{i}/{len(all_images)}] Processing: {img_path.name}")
                self.extract_features(img_path, use_cache=True)
        
        print(f"[DINOv3] Dataset pre-loading complete")


# Singleton instance for reuse
_extractor_instance = None

def get_extractor(model_size: str = 'small', cache_dir: Optional[str] = None) -> DINOv3Extractor:
    """Get or create a singleton DINOv3 extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = DINOv3Extractor(model_size=model_size, cache_dir=cache_dir)
    return _extractor_instance
