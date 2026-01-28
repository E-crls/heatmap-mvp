"""
Experiment 1: DINOv3 Real Implementation
Uses the actual DINOv3 from Hugging Face Transformers (released 2025)

Key insight: The current technique compares patches by POSITION (patch 1 of image A vs patch 1 of image B).
This doesn't work when images have different compositions (e.g., full body vs head only).

Better approach: Cross-similarity - find which patches in the query are similar to ANY patch in the target.
This shows what SEMANTIC regions match, not positional regions.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional

# Check if transformers has DINOv3
try:
    from transformers import AutoImageProcessor, AutoModel
    HAS_TRANSFORMERS_DINOV3 = True
except ImportError:
    HAS_TRANSFORMERS_DINOV3 = False


class DINOv3Extractor:
    """
    DINOv3 feature extractor using Hugging Face Transformers.
    Model: facebook/dinov3-vits16-pretrain-lvd1689m (or larger variants)
    """
    
    AVAILABLE_MODELS = {
        'small': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    }
    
    def __init__(self, model_size: str = 'small'):
        if not HAS_TRANSFORMERS_DINOV3:
            raise ImportError("transformers library not found or doesn't have DINOv3 support")
        
        self.model_name = self.AVAILABLE_MODELS.get(model_size, self.AVAILABLE_MODELS['small'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[DINOv3] Loading {self.model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 0)
        print(f"[DINOv3] Loaded. Patch size: {self.patch_size}, Register tokens: {self.num_register_tokens}")
    
    def extract_features(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Extract CLS token and patch features.
        
        Returns:
            cls_token: [hidden_size]
            patch_features: [num_patches, hidden_size]
            grid_shape: (height_patches, width_patches)
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state  # [1, 1 + num_register + num_patches, hidden_size]
        
        # Extract components
        cls_token = last_hidden_state[:, 0, :].cpu().numpy().flatten()
        
        # Skip CLS and register tokens
        patch_features = last_hidden_state[:, 1 + self.num_register_tokens:, :].cpu().numpy()
        patch_features = patch_features[0]  # Remove batch dim
        
        # Calculate grid shape
        _, _, img_height, img_width = inputs.pixel_values.shape
        grid_h = img_height // self.patch_size
        grid_w = img_width // self.patch_size
        
        return cls_token, patch_features, (grid_h, grid_w)


class CrossSimilarityHeatmap:
    """
    Generate heatmap using cross-similarity between patches.
    
    Instead of comparing patch_i(A) vs patch_i(B), we compute:
    - For each patch in the query: find the MAX similarity to ANY patch in the target
    
    This answers: "Which regions in my query are semantically similar to something in the target?"
    """
    
    @staticmethod
    def compute_max_similarity(query_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
        """
        For each query patch, find the maximum similarity to any target patch.
        
        Args:
            query_patches: [num_patches, hidden_size]
            target_patches: [num_patches, hidden_size]
        
        Returns:
            max_similarities: [num_patches] - max similarity for each query patch
        """
        # Compute full similarity matrix: [query_patches, target_patches]
        similarity_matrix = cosine_similarity(query_patches, target_patches)
        
        # For each query patch, take the max similarity to any target patch
        max_similarities = similarity_matrix.max(axis=1)
        
        return max_similarities
    
    @staticmethod
    def compute_mean_top_k_similarity(query_patches: np.ndarray, target_patches: np.ndarray, k: int = 5) -> np.ndarray:
        """
        For each query patch, compute mean of top-K similar target patches.
        More robust than just max.
        """
        similarity_matrix = cosine_similarity(query_patches, target_patches)
        
        # Sort each row and take top-k mean
        top_k_sims = np.sort(similarity_matrix, axis=1)[:, -k:]
        mean_top_k = top_k_sims.mean(axis=1)
        
        return mean_top_k
    
    @staticmethod
    def generate_heatmap(
        query_image_path: Path,
        query_patches: np.ndarray,
        target_patches: np.ndarray,
        grid_shape: Tuple[int, int],
        method: str = 'max',
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Generate heatmap overlay on query image.
        
        Args:
            query_image_path: Path to query image
            query_patches: Query patch embeddings
            target_patches: Target patch embeddings
            grid_shape: (grid_h, grid_w)
            method: 'max' or 'top_k'
            alpha: Overlay opacity
        """
        # Compute similarity
        if method == 'max':
            similarities = CrossSimilarityHeatmap.compute_max_similarity(query_patches, target_patches)
        else:
            similarities = CrossSimilarityHeatmap.compute_mean_top_k_similarity(query_patches, target_patches)
        
        # Reshape to grid
        grid_h, grid_w = grid_shape
        similarity_grid = similarities.reshape(grid_h, grid_w)
        
        print(f"  Similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
        
        # Load query image
        original = Image.open(query_image_path).convert('RGB')
        img_array = np.array(original)
        
        # Normalize to 0-255
        sim_norm = ((similarity_grid - similarity_grid.min()) /
                    (similarity_grid.max() - similarity_grid.min() + 1e-8) * 255).astype(np.uint8)
        
        # Resize to image size
        heatmap = cv2.resize(sim_norm, (img_array.shape[1], img_array.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
        
        # Apply JET colormap (blue=low, red=high)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlay)


def test_dinov3():
    """Test DINOv3 extraction"""
    print("=" * 60)
    print("Testing DINOv3 Extractor")
    print("=" * 60)
    
    try:
        extractor = DINOv3Extractor(model_size='small')
        
        # Load test image
        test_image = Path(__file__).parent.parent / "static" / "images" / "input_image.png"
        if test_image.exists():
            image = Image.open(test_image).convert('RGB')
            cls_token, patches, grid_shape = extractor.extract_features(image)
            
            print(f"\nResults for {test_image.name}:")
            print(f"  CLS token shape: {cls_token.shape}")
            print(f"  Patch features shape: {patches.shape}")
            print(f"  Grid shape: {grid_shape}")
            print("\n✓ DINOv3 extraction successful!")
        else:
            print(f"Test image not found: {test_image}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nDINOv3 may not be available yet. Try: pip install transformers --upgrade")


if __name__ == "__main__":
    test_dinov3()
