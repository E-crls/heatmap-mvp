"""
Experiment 2: Cross-Similarity Heatmap using DINOv2
Uses the existing DINOv2 model but with CORRECT cross-similarity technique.

PROBLEM WITH CURRENT APPROACH:
- Current: Compare patch_i(query) vs patch_i(target) - positional comparison
- This fails when images have different layouts (e.g., head vs full body)

CORRECT APPROACH:
- For each patch in query, find the MAXIMUM similarity to ANY patch in target
- This shows: "Which parts of my query image have semantic matches in the target?"

Example: If query is a bull HEAD and target is a full bull BODY:
- The horn patches in query should match horn patches in target (wherever they are)
- The eye patches in query should match eye patches in target
- Background patches should NOT match anything specific → low similarity
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List

# Paths
BASE_DIR = Path(__file__).parent.parent.parent  # d:\INPI\heatmap
CACHE_DIR = BASE_DIR / "embeddings_cache"
IMAGES_DIR = Path(__file__).parent.parent / "static" / "images"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_embeddings(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings"""
    cache_file = CACHE_DIR / f"{name}_dinov3.npz"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    data = np.load(cache_file)
    return data['cls_embedding'].flatten(), data['patch_embeddings'][0]


def cross_similarity_max(query_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
    """
    For each query patch, find MAX similarity to ANY target patch.
    
    This answers: "For each region of my query, is there a similar region ANYWHERE in the target?"
    """
    # Full similarity matrix: [num_query_patches, num_target_patches]
    sim_matrix = cosine_similarity(query_patches, target_patches)
    
    # For each query patch, take the max across all target patches
    max_sim = sim_matrix.max(axis=1)
    
    return max_sim


def cross_similarity_topk_mean(query_patches: np.ndarray, target_patches: np.ndarray, k: int = 3) -> np.ndarray:
    """
    For each query patch, compute mean of top-K similar target patches.
    More stable than just max.
    """
    sim_matrix = cosine_similarity(query_patches, target_patches)
    
    # Sort each row and take top-k mean
    sorted_sims = np.sort(sim_matrix, axis=1)[:, -k:]
    mean_topk = sorted_sims.mean(axis=1)
    
    return mean_topk


def positional_similarity(query_patches: np.ndarray, target_patches: np.ndarray) -> np.ndarray:
    """
    Original (WRONG) approach: Compare patch by position.
    This only works if images have identical composition/layout.
    """
    similarities = []
    for i in range(query_patches.shape[0]):
        sim = cosine_similarity(query_patches[i:i+1], target_patches[i:i+1])[0][0]
        similarities.append(sim)
    return np.array(similarities)


def generate_heatmap(
    image_path: Path,
    similarity_vector: np.ndarray,
    output_path: Path,
    alpha: float = 0.5,
    title: str = ""
) -> None:
    """Generate and save heatmap overlay"""
    # Reshape to grid (37x37 for DINOv2 with 518x518 input)
    grid_size = int(np.sqrt(len(similarity_vector)))
    similarity_grid = similarity_vector.reshape(grid_size, grid_size)
    
    # Load image
    original = Image.open(image_path).convert('RGB')
    img_array = np.array(original)
    
    # Normalize to 0-255
    sim_norm = ((similarity_grid - similarity_grid.min()) /
                (similarity_grid.max() - similarity_grid.min() + 1e-8) * 255).astype(np.uint8)
    
    # Resize
    heatmap = cv2.resize(sim_norm, (img_array.shape[1], img_array.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    
    # Apply colormap (JET: blue=low, red=high)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(img_array, 1-alpha, heatmap_colored, alpha, 0)
    
    # Save
    Image.fromarray(overlay).save(output_path, quality=95)
    print(f"  Saved: {output_path.name}")


def compare_methods():
    """Compare positional vs cross-similarity methods"""
    print("=" * 70)
    print("EXPERIMENT 2: CROSS-SIMILARITY HEATMAP")
    print("=" * 70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    query_cls, query_patches = load_embeddings("input_image")
    target_cls, target_patches = load_embeddings("istockphoto-1272809203-612x612")
    
    print(f"  Query patches: {query_patches.shape}")
    print(f"  Target patches: {target_patches.shape}")
    
    # Find query image
    query_image = IMAGES_DIR / "input_image.png"
    if not query_image.exists():
        query_image = BASE_DIR / "data" / "input" / "input_image.png"
    
    print(f"  Query image: {query_image}")
    
    # Method 1: Positional (current - WRONG)
    print("\n[Method 1] POSITIONAL similarity (current implementation):")
    pos_sim = positional_similarity(query_patches, target_patches)
    print(f"  Stats: min={pos_sim.min():.4f}, max={pos_sim.max():.4f}, mean={pos_sim.mean():.4f}")
    generate_heatmap(query_image, pos_sim, OUTPUT_DIR / "method1_positional.png")
    
    # Method 2: Cross-similarity MAX
    print("\n[Method 2] CROSS-SIMILARITY (max):")
    cross_max = cross_similarity_max(query_patches, target_patches)
    print(f"  Stats: min={cross_max.min():.4f}, max={cross_max.max():.4f}, mean={cross_max.mean():.4f}")
    generate_heatmap(query_image, cross_max, OUTPUT_DIR / "method2_cross_max.png")
    
    # Method 3: Cross-similarity Top-3 Mean
    print("\n[Method 3] CROSS-SIMILARITY (top-3 mean):")
    cross_topk = cross_similarity_topk_mean(query_patches, target_patches, k=3)
    print(f"  Stats: min={cross_topk.min():.4f}, max={cross_topk.max():.4f}, mean={cross_topk.mean():.4f}")
    generate_heatmap(query_image, cross_topk, OUTPUT_DIR / "method3_cross_topk.png")
    
    # Method 4: Cross-similarity Top-5 Mean  
    print("\n[Method 4] CROSS-SIMILARITY (top-5 mean):")
    cross_top5 = cross_similarity_topk_mean(query_patches, target_patches, k=5)
    print(f"  Stats: min={cross_top5.min():.4f}, max={cross_top5.max():.4f}, mean={cross_top5.mean():.4f}")
    generate_heatmap(query_image, cross_top5, OUTPUT_DIR / "method4_cross_top5.png")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print(f"Check outputs in: {OUTPUT_DIR}")
    print("")
    print("EXPECTED RESULTS:")
    print("  - Method 1 (positional): Poor - compares wrong regions")
    print("  - Method 2-4 (cross): Better - finds semantic matches")
    print("  - Horn/head regions should be RED (high similarity)")
    print("  - Background should be BLUE (low similarity)")
    print("=" * 70)


def test_with_second_similar():
    """Also test with the second most similar image"""
    print("\n" + "=" * 70)
    print("TESTING WITH SECOND SIMILAR IMAGE (e2e4662e)")
    print("=" * 70)
    
    query_cls, query_patches = load_embeddings("input_image")
    target_cls, target_patches = load_embeddings("e2e4662e541a44a55410f40315f9222d")
    
    query_image = IMAGES_DIR / "input_image.png"
    if not query_image.exists():
        query_image = BASE_DIR / "data" / "input" / "input_image.png"
    
    # Cross-similarity max
    print("\n[Method 2] CROSS-SIMILARITY (max):")
    cross_max = cross_similarity_max(query_patches, target_patches)
    print(f"  Stats: min={cross_max.min():.4f}, max={cross_max.max():.4f}, mean={cross_max.mean():.4f}")
    generate_heatmap(query_image, cross_max, OUTPUT_DIR / "method2_cross_max_e2e.png")
    
    print("\nDone!")


if __name__ == "__main__":
    compare_methods()
    test_with_second_similar()
