"""
Heatmap Generator for INPI Visual Search MVP
Generates patch-wise similarity heatmaps with transparency threshold support
"""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Optional
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64


class HeatmapGenerator:
    """
    Generates similarity heatmaps by comparing patch embeddings between images.
    Supports transparency thresholds and multi-heatmap overlays.
    """
    
    # Default colormap for similarity visualization
    # Red = high similarity, Blue = low similarity
    DEFAULT_COLORMAP = cv2.COLORMAP_JET
    
    # Grid size for DINOv3 with 518x518 input (patch size 14x14)
    DEFAULT_GRID_SIZE = 37
    
    def __init__(self, threshold: float = 0.5, alpha: float = 0.6):
        """
        Initialize the heatmap generator.
        
        Args:
            threshold: Intensity threshold (0-1). Pixels below become transparent.
            alpha: Default overlay opacity (0-1).
        """
        self.threshold = threshold
        self.alpha = alpha
    
    def compute_patch_similarity(self, 
                                  input_patches: np.ndarray, 
                                  target_patches: np.ndarray) -> np.ndarray:
        """
        Compute patch-wise cosine similarity between two sets of patch embeddings.
        
        Args:
            input_patches: [num_patches, feature_dim]
            target_patches: [num_patches, feature_dim]
        
        Returns:
            similarity_vector: [num_patches] similarity scores
        """
        assert input_patches.shape == target_patches.shape, \
            f"Patch shape mismatch: {input_patches.shape} vs {target_patches.shape}"
        
        # Compute cosine similarity for each patch pair
        similarities = []
        for i in range(input_patches.shape[0]):
            sim = cosine_similarity(
                input_patches[i:i+1],
                target_patches[i:i+1]
            )[0][0]
            similarities.append(sim)
        
        return np.array(similarities)
    
    def reshape_to_grid(self, similarity_vector: np.ndarray, 
                        grid_size: Optional[int] = None) -> np.ndarray:
        """
        Reshape 1D similarity vector to 2D spatial grid.
        
        Args:
            similarity_vector: [num_patches] 
            grid_size: Expected grid size (auto-computed if None)
        
        Returns:
            similarity_grid: [grid_size, grid_size]
        """
        if grid_size is None:
            grid_size = int(np.sqrt(len(similarity_vector)))
        
        return similarity_vector.reshape(grid_size, grid_size)
    
    def apply_threshold(self, 
                        heatmap: np.ndarray, 
                        threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply threshold to heatmap, creating transparency mask.
        
        Args:
            heatmap: Normalized heatmap (0-1)
            threshold: Intensity threshold. Uses instance default if None.
        
        Returns:
            Tuple of (thresholded_heatmap, alpha_mask)
            - thresholded_heatmap: Values below threshold set to 0
            - alpha_mask: Binary mask (1 where above threshold, 0 below)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Normalize if needed
        if heatmap.max() > 1.0:
            heatmap = heatmap / 255.0
        
        # Create alpha mask
        alpha_mask = (heatmap >= threshold).astype(np.float32)
        
        # Apply threshold
        thresholded = np.where(heatmap >= threshold, heatmap, 0)
        
        return thresholded, alpha_mask
    
    def create_heatmap_rgba(self,
                            similarity_grid: np.ndarray,
                            target_size: Tuple[int, int],
                            threshold: Optional[float] = None,
                            colormap: int = None) -> np.ndarray:
        """
        Create RGBA heatmap with transparency for low intensity regions.
        
        Args:
            similarity_grid: 2D similarity map [H, W]
            target_size: (width, height) to resize heatmap
            threshold: Intensity threshold
            colormap: OpenCV colormap
        
        Returns:
            heatmap_rgba: [height, width, 4] RGBA image
        """
        if colormap is None:
            colormap = self.DEFAULT_COLORMAP
        if threshold is None:
            threshold = self.threshold
        
        # Normalize to 0-1
        sim_norm = (similarity_grid - similarity_grid.min()) / \
                   (similarity_grid.max() - similarity_grid.min() + 1e-8)
        
        # Apply threshold to get alpha mask
        _, alpha_mask = self.apply_threshold(sim_norm, threshold)
        
        # Scale to 0-255 for colormap
        sim_uint8 = (sim_norm * 255).astype(np.uint8)
        
        # Resize to target size
        sim_resized = cv2.resize(sim_uint8, target_size, interpolation=cv2.INTER_CUBIC)
        alpha_resized = cv2.resize(alpha_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Apply colormap (returns BGR)
        heatmap_bgr = cv2.applyColorMap(sim_resized, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Create RGBA with alpha channel
        heatmap_rgba = np.zeros((heatmap_rgb.shape[0], heatmap_rgb.shape[1], 4), dtype=np.uint8)
        heatmap_rgba[:, :, :3] = heatmap_rgb
        heatmap_rgba[:, :, 3] = (alpha_resized * 255).astype(np.uint8)
        
        return heatmap_rgba
    
    def overlay_single_simple(self,
                       base_image: Union[str, Path, Image.Image],
                       similarity_grid: np.ndarray,
                       alpha: float = 0.5,
                       colormap: int = None) -> Image.Image:
        """
        Overlay a single heatmap on the base image using simple blending.
        This is the correct technique from the original step3_generate_heatmap.py.
        
        Args:
            base_image: Original image
            similarity_grid: 2D similarity map
            alpha: Overlay opacity (0-1), default 0.5
            colormap: OpenCV colormap
        
        Returns:
            Overlaid PIL Image
        """
        if colormap is None:
            colormap = self.DEFAULT_COLORMAP
        
        # Load base image if needed
        if isinstance(base_image, (str, Path)):
            original = Image.open(base_image).convert('RGB')
        else:
            original = base_image.convert('RGB')
        
        img_array = np.array(original)
        
        # Normalize similarity to 0-255
        sim_norm = ((similarity_grid - similarity_grid.min()) /
                    (similarity_grid.max() - similarity_grid.min() + 1e-8) * 255).astype(np.uint8)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(sim_norm, (img_array.shape[1], img_array.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image using cv2.addWeighted (the correct technique)
        overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
        
        # Convert to PIL Image
        result = Image.fromarray(overlay)
        return result
    
    def overlay_single(self,
                       base_image: Union[str, Path, Image.Image],
                       similarity_grid: np.ndarray,
                       alpha: Optional[float] = None,
                       threshold: Optional[float] = None) -> Image.Image:
        """
        Overlay a single heatmap on the base image.
        Now uses the simple blending technique from the original script.
        
        Args:
            base_image: Original image
            similarity_grid: 2D similarity map
            alpha: Overlay opacity (default 0.5)
            threshold: Ignored (kept for compatibility)
        
        Returns:
            Overlaid PIL Image
        """
        if alpha is None:
            alpha = 0.5  # Default to 50% opacity
        
        return self.overlay_single_simple(base_image, similarity_grid, alpha)
    
    def overlay_multiple(self,
                         base_image: Union[str, Path, Image.Image],
                         similarity_grids: List[np.ndarray],
                         alphas: Optional[List[float]] = None,
                         threshold: Optional[float] = None) -> Image.Image:
        """
        Overlay multiple heatmaps on the base image using simple blending.
        Higher similarity images get higher opacity by default.
        
        Args:
            base_image: Original image
            similarity_grids: List of 2D similarity maps (sorted by similarity, highest first)
            alphas: List of opacity values for each heatmap
            threshold: Ignored (kept for compatibility)
        
        Returns:
            Overlaid PIL Image with all heatmaps
        """
        # Default alphas: decreasing opacity for less similar images
        if alphas is None:
            # Most similar gets 0.5, then 0.35, then 0.2
            default_alphas = [0.5, 0.35, 0.2]
            alphas = default_alphas[:len(similarity_grids)]
        
        # Load base image
        if isinstance(base_image, (str, Path)):
            result = np.array(Image.open(base_image).convert('RGB'))
        else:
            result = np.array(base_image.convert('RGB'))
        
        # Apply heatmaps in order (most similar last so it's most visible)
        for i in range(len(similarity_grids) - 1, -1, -1):
            similarity_grid = similarity_grids[i]
            alpha = alphas[i] if i < len(alphas) else 0.3
            
            # Normalize similarity to 0-255
            sim_norm = ((similarity_grid - similarity_grid.min()) /
                        (similarity_grid.max() - similarity_grid.min() + 1e-8) * 255).astype(np.uint8)
            
            # Resize heatmap to match image size
            heatmap = cv2.resize(sim_norm, (result.shape[1], result.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, self.DEFAULT_COLORMAP)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend with current result
            result = cv2.addWeighted(result, 1-alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(result)
    
    def generate_heatmap(self,
                         input_patches: np.ndarray,
                         target_patches: np.ndarray,
                         base_image: Union[str, Path, Image.Image],
                         alpha: Optional[float] = None) -> Tuple[Image.Image, float]:
        """
        Complete pipeline: compute similarity and generate overlay.
        Uses simple 50% blend technique from original script.
        
        Args:
            input_patches: Patch embeddings from input image
            target_patches: Patch embeddings from target image
            base_image: Image to overlay heatmap on
            alpha: Overlay opacity (default 0.5)
        
        Returns:
            Tuple of (overlay_image, average_similarity)
        """
        if alpha is None:
            alpha = 0.5
            
        # Compute patch-wise similarity
        similarity_vector = self.compute_patch_similarity(input_patches, target_patches)
        similarity_grid = self.reshape_to_grid(similarity_vector)
        
        # Compute average similarity
        avg_similarity = float(similarity_vector.mean())
        
        # Generate overlay using simple blending
        overlay = self.overlay_single_simple(base_image, similarity_grid, alpha)
        
        return overlay, avg_similarity
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            format: Output format (PNG, JPEG, etc.)
        
        Returns:
            Base64 encoded string with data URI prefix
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        mime = 'image/png' if format.upper() == 'PNG' else 'image/jpeg'
        
        return f"data:{mime};base64,{b64}"
    
    def save_heatmap(self, 
                     image: Image.Image, 
                     output_path: Union[str, Path],
                     quality: int = 95):
        """
        Save heatmap image to file.
        
        Args:
            image: PIL Image to save
            output_path: Destination path
            quality: JPEG quality (ignored for PNG)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            # Convert RGBA to RGB for JPEG
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])
                rgb_image.save(output_path, quality=quality)
            else:
                image.save(output_path, quality=quality)
        else:
            image.save(output_path)


# Singleton instance
_heatmap_generator = None

def get_heatmap_generator(threshold: float = 0.5, alpha: float = 0.6) -> HeatmapGenerator:
    """Get or create singleton heatmap generator instance."""
    global _heatmap_generator
    if _heatmap_generator is None:
        _heatmap_generator = HeatmapGenerator(threshold=threshold, alpha=alpha)
    return _heatmap_generator
