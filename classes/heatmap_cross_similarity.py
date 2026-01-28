"""
Heatmap Generator using Cross-Similarity technique.

CORRECT APPROACH:
For each patch in the query image, find the MAXIMUM similarity to ANY patch in the target.
This shows: "Which parts of my query have semantic matches ANYWHERE in the target?"

This is fundamentally different from positional comparison (patch_i vs patch_i),
which only works when images have identical composition/layout.
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Optional
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64


class CrossSimilarityHeatmap:
    """
    Generates similarity heatmaps using cross-similarity between patches.
    """
    
    # COLORMAP_JET: blue=low similarity, red=high similarity
    DEFAULT_COLORMAP = cv2.COLORMAP_JET
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize the heatmap generator.
        
        Args:
            alpha: Default overlay opacity (0-1).
        """
        self.alpha = alpha
    
    def compute_cross_similarity_max(self, 
                                      query_patches: np.ndarray, 
                                      target_patches: np.ndarray) -> np.ndarray:
        """
        For each query patch, find the MAX similarity to ANY target patch.
        
        This answers: "For each region of my query, is there a similar region ANYWHERE in the target?"
        
        Args:
            query_patches: [num_patches, hidden_size] - patches from the query image
            target_patches: [num_patches, hidden_size] - patches from the target image
        
        Returns:
            max_similarities: [num_patches] - max similarity for each query patch
        """
        # Compute full similarity matrix: [num_query_patches, num_target_patches]
        similarity_matrix = cosine_similarity(query_patches, target_patches)
        
        # For each query patch, take the max across all target patches
        max_similarities = similarity_matrix.max(axis=1)
        
        return max_similarities
    
    def compute_cross_similarity_topk(self, 
                                       query_patches: np.ndarray, 
                                       target_patches: np.ndarray,
                                       k: int = 3) -> np.ndarray:
        """
        For each query patch, compute mean of top-K similar target patches.
        More robust than just max (less sensitive to outliers).
        
        Args:
            query_patches: [num_patches, hidden_size]
            target_patches: [num_patches, hidden_size]
            k: Number of top matches to average
        
        Returns:
            mean_topk_similarities: [num_patches]
        """
        similarity_matrix = cosine_similarity(query_patches, target_patches)
        
        # Sort each row and take top-k mean
        sorted_sims = np.sort(similarity_matrix, axis=1)[:, -k:]
        mean_topk = sorted_sims.mean(axis=1)
        
        return mean_topk
    
    def generate_heatmap(self,
                         query_patches: np.ndarray,
                         target_patches: np.ndarray,
                         query_image: Union[str, Path, Image.Image],
                         grid_shape: Tuple[int, int],
                         method: str = 'max',
                         alpha: Optional[float] = None) -> Tuple[Image.Image, float]:
        """
        Generate heatmap overlay on the query image.
        
        The heatmap shows which regions of the QUERY are similar to something in the TARGET.
        Red = high similarity (this part of query matches something in target)
        Blue = low similarity (this part of query doesn't match anything in target)
        
        Args:
            query_patches: Patch embeddings from query image
            target_patches: Patch embeddings from target image
            query_image: The query image (heatmap is overlaid on this)
            grid_shape: (grid_h, grid_w) - spatial dimensions of patches
            method: 'max' or 'topk'
            alpha: Overlay opacity (default 0.5)
        
        Returns:
            Tuple of (overlay_image, average_similarity)
        """
        if alpha is None:
            alpha = self.alpha
        
        # Compute cross-similarity
        if method == 'max':
            similarities = self.compute_cross_similarity_max(query_patches, target_patches)
        else:
            similarities = self.compute_cross_similarity_topk(query_patches, target_patches)
        
        avg_similarity = float(similarities.mean())
        
        # Reshape to grid
        grid_h, grid_w = grid_shape
        similarity_grid = similarities.reshape(grid_h, grid_w)
        
        # Load query image
        if isinstance(query_image, (str, Path)):
            original = Image.open(query_image).convert('RGB')
        else:
            original = query_image.convert('RGB')
        
        img_array = np.array(original)
        
        # Normalize similarity to 0-255
        sim_min, sim_max = similarity_grid.min(), similarity_grid.max()
        sim_norm = ((similarity_grid - sim_min) / (sim_max - sim_min + 1e-8) * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(sim_norm, (img_array.shape[1], img_array.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap (JET: blue=low, red=high)
        heatmap_colored = cv2.applyColorMap(heatmap, self.DEFAULT_COLORMAP)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlay), avg_similarity
    
    def generate_multiple_heatmaps(self,
                                   query_patches: np.ndarray,
                                   target_patches_list: List[np.ndarray],
                                   query_image: Union[str, Path, Image.Image],
                                   grid_shape: Tuple[int, int],
                                   method: str = 'max',
                                   alphas: Optional[List[float]] = None) -> Image.Image:
        """
        Generate combined heatmap from multiple targets.
        Each target contributes to the final heatmap with decreasing opacity.
        
        Args:
            query_patches: Patch embeddings from query image
            target_patches_list: List of patch embeddings from target images (sorted by similarity)
            query_image: The query image
            grid_shape: (grid_h, grid_w)
            method: 'max' or 'topk'
            alphas: List of opacity values for each heatmap
        
        Returns:
            Combined overlay image
        """
        if alphas is None:
            alphas = [0.5, 0.35, 0.2]  # Decreasing opacity for less similar images
        
        # Load query image
        if isinstance(query_image, (str, Path)):
            result = np.array(Image.open(query_image).convert('RGB'))
        else:
            result = np.array(query_image.convert('RGB'))
        
        grid_h, grid_w = grid_shape
        
        # Apply heatmaps in reverse order (most similar last so it's most visible)
        for i in range(len(target_patches_list) - 1, -1, -1):
            target_patches = target_patches_list[i]
            alpha = alphas[i] if i < len(alphas) else 0.3
            
            # Compute cross-similarity
            if method == 'max':
                similarities = self.compute_cross_similarity_max(query_patches, target_patches)
            else:
                similarities = self.compute_cross_similarity_topk(query_patches, target_patches)
            
            # Reshape to grid
            similarity_grid = similarities.reshape(grid_h, grid_w)
            
            # Normalize
            sim_min, sim_max = similarity_grid.min(), similarity_grid.max()
            sim_norm = ((similarity_grid - sim_min) / (sim_max - sim_min + 1e-8) * 255).astype(np.uint8)
            
            # Resize
            heatmap = cv2.resize(sim_norm, (result.shape[1], result.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, self.DEFAULT_COLORMAP)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend
            result = cv2.addWeighted(result, 1-alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(result)
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string with data URI prefix."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime = 'image/png' if format.upper() == 'PNG' else 'image/jpeg'
        return f"data:{mime};base64,{b64}"
