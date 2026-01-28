"""
DINOv3 Feature Extractor for INPI Visual Search MVP
Supports both CLS token (for similarity search) and patch embeddings (for heatmaps)
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import os
import hashlib


class DINOv3Extractor:
    """
    DINOv3-based feature extractor with caching support.
    Extracts both CLS token and patch embeddings for heatmap generation.
    """
    
    def __init__(self, model_size: str = 'large', cache_dir: Optional[str] = None):
        """
        Initialize the DINOv3 extractor.
        
        Args:
            model_size: Model size ('small', 'base', 'large', 'giant')
            cache_dir: Directory to store embedding cache. Uses env var or default.
        """
        self.model_size = model_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use environment variable or default path
            base_dir = Path(__file__).parent.parent.parent
            self.cache_dir = Path(os.getenv('EMBEDDINGS_CACHE_DIR', base_dir / 'embeddings_cache'))
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load DINOv3 model from torch hub"""
        print(f"[DINOv3] Loading model ({self.model_size}) on {self.device}...")
        
        model_name = f'dinov2_vit{self.model_size[0]}14'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Setup transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"[DINOv3] Model loaded successfully")
    
    def _get_cache_key(self, image_path: Union[str, Path]) -> str:
        """Generate a cache key from image path"""
        path = Path(image_path)
        return f"{path.stem}_dinov3.npz"
    
    def _get_cache_path(self, image_path: Union[str, Path]) -> Path:
        """Get full cache file path for an image"""
        return self.cache_dir / self._get_cache_key(image_path)
    
    def is_cached(self, image_path: Union[str, Path]) -> bool:
        """Check if embeddings are cached for an image"""
        return self._get_cache_path(image_path).exists()
    
    def extract_features(self, image: Union[str, Path, Image.Image], 
                        use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract DINOv3 features from an image.
        
        Args:
            image: Image path or PIL Image
            use_cache: Whether to use/save cache (only works with paths)
        
        Returns:
            Tuple of (cls_embedding, patch_embeddings)
            - cls_embedding: [1, feature_dim] - for similarity search
            - patch_embeddings: [1, num_patches, feature_dim] - for heatmaps
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
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.forward_features(image_tensor)
            cls_token = features['x_norm_clstoken']
            patch_tokens = features['x_norm_patchtokens']
        
        cls_embedding = cls_token.cpu().numpy()
        patch_embeddings = patch_tokens.cpu().numpy()
        
        # Save to cache if we have a path
        if image_path and use_cache:
            self._save_to_cache(image_path, cls_embedding, patch_embeddings)
        
        return cls_embedding, patch_embeddings
    
    def _load_from_cache(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings from cache"""
        cache_path = self._get_cache_path(image_path)
        data = np.load(cache_path)
        return data['cls_embedding'], data['patch_embeddings']
    
    def _save_to_cache(self, image_path: Union[str, Path], 
                       cls_embedding: np.ndarray, 
                       patch_embeddings: np.ndarray):
        """Save embeddings to cache"""
        cache_path = self._get_cache_path(image_path)
        np.savez_compressed(
            cache_path,
            cls_embedding=cls_embedding,
            patch_embeddings=patch_embeddings,
            image_path=str(image_path)
        )
    
    def get_cls_embedding(self, image: Union[str, Path, Image.Image], 
                          use_cache: bool = True) -> np.ndarray:
        """
        Get only the CLS embedding (for similarity search).
        
        Returns:
            cls_embedding: [feature_dim] - flattened for vector search
        """
        cls_emb, _ = self.extract_features(image, use_cache)
        return cls_emb.flatten()
    
    def get_patch_embeddings(self, image: Union[str, Path, Image.Image],
                             use_cache: bool = True) -> np.ndarray:
        """
        Get only the patch embeddings (for heatmap generation).
        
        Returns:
            patch_embeddings: [num_patches, feature_dim]
        """
        _, patch_emb = self.extract_features(image, use_cache)
        return patch_emb[0]  # Remove batch dimension
    
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
        
        Args:
            image_dir: Directory containing images
            extensions: Image file extensions to process
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

def get_extractor(model_size: str = 'large', cache_dir: Optional[str] = None) -> DINOv3Extractor:
    """
    Get or create a singleton DINOv3 extractor instance.
    
    Args:
        model_size: Model size (only used on first call)
        cache_dir: Cache directory (only used on first call)
    
    Returns:
        DINOv3Extractor instance
    """
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = DINOv3Extractor(model_size=model_size, cache_dir=cache_dir)
    return _extractor_instance
