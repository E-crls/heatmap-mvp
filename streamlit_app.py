"""
INPI Heatmap Visual MVP - Streamlit Version
Uses DINOv3 (real) + Cross-Similarity for accurate heatmaps
Optimized for Streamlit Cloud deployment
"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import base64
import io
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from settings.config import Config
from classes.dinov3_extractor_real import DINOv3Extractor
from classes.heatmap_cross_similarity import CrossSimilarityHeatmap

# Page config
st.set_page_config(
    page_title="INPI Visual Search - Heatmap Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #005c9e 0%, #0077cc 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        transition: all 0.3s;
    }
    .image-container:hover {
        border-color: #005c9e;
        box-shadow: 0 4px 12px rgba(0, 92, 158, 0.2);
    }
    .similarity-score {
        background: linear-gradient(135deg, #fbbf24 0%, #f97316 50%, #ef4444 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .step-badge {
        background: #005c9e;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Imagens selecionáveis
SELECTABLE_IMAGES = ["03_chicago_bulls", "input_image", "Audi_logo_detail.svg"]


@st.cache_resource
def load_extractor():
    """Load DINOv3 extractor (cached)"""
    return DINOv3Extractor(
        model_size='small',
        cache_dir=str(Config.CACHE_DIR)
    )


@st.cache_resource
def load_heatmap_generator():
    """Load heatmap generator (cached)"""
    return CrossSimilarityHeatmap(alpha=0.5)


def get_selectable_images():
    """Get list of images user can select"""
    result = set()
    for ext in Config.IMAGE_EXTENSIONS:
        for img in Config.IMAGES_DIR.glob(f"*.{ext}"):
            if img.stem in SELECTABLE_IMAGES:
                result.add(img)
        for img in Config.IMAGES_DIR.glob(f"*.{ext.upper()}"):
            if img.stem in SELECTABLE_IMAGES:
                result.add(img)
    return sorted(list(result), key=lambda x: x.stem)


def get_all_images():
    """Get all images in dataset"""
    result = set()
    for ext in Config.IMAGE_EXTENSIONS:
        for img in Config.IMAGES_DIR.glob(f"*.{ext}"):
            result.add(img)
        for img in Config.IMAGES_DIR.glob(f"*.{ext.upper()}"):
            result.add(img)
    return list(result)


def find_similar_images(extractor, query_path, all_images, top_k=5):
    """Find similar images to query"""
    query_path = Path(query_path)
    results = []
    
    for img_path in all_images:
        if img_path.resolve() == query_path.resolve():
            continue
        
        similarity = extractor.compute_similarity(query_path, img_path)
        results.append({
            'path': img_path,
            'name': img_path.stem,
            'similarity': similarity
        })
    
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def generate_heatmap(extractor, heatmap_gen, query_path, target_path):
    """Generate heatmap between query and target"""
    query_patches, query_grid = extractor.get_patch_embeddings(query_path)
    target_patches, _ = extractor.get_patch_embeddings(target_path)
    
    # Generate overlay (heatmap blended with original)
    query_img = Image.open(query_path).convert('RGB')
    overlay_img, avg_similarity = heatmap_gen.generate_heatmap(
        query_patches=query_patches,
        target_patches=target_patches,
        query_image=query_img,
        grid_shape=query_grid,
        method='max'
    )
    
    # Generate pure heatmap (without original image)
    cross_sim = heatmap_gen.compute_cross_similarity_max(query_patches, target_patches)
    grid_h, grid_w = query_grid
    similarity_grid = cross_sim.reshape(grid_h, grid_w)
    
    # Normalize and create heatmap
    import cv2
    sim_min, sim_max = similarity_grid.min(), similarity_grid.max()
    sim_norm = ((similarity_grid - sim_min) / (sim_max - sim_min + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.resize(sim_norm, (query_img.width, query_img.height), interpolation=cv2.INTER_CUBIC)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_img = Image.fromarray(heatmap_colored)
    
    return heatmap_img, overlay_img, avg_similarity


def image_to_base64(img):
    """Convert PIL image to base64"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 INPI Visual Search - Heatmap Demo</h1>
    <p>Demonstração de Mapa de Calor para Similaridade de Marcas usando DINOv3</p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Carregando modelo DINOv3..."):
    extractor = load_extractor()
    heatmap_gen = load_heatmap_generator()

# Step 1: Image Selection
st.markdown("### <span class='step-badge'>1</span> Selecione uma Imagem", unsafe_allow_html=True)
st.write("Escolha uma das imagens abaixo para encontrar marcas similares e visualizar o mapa de calor.")

selectable_images = get_selectable_images()
all_images = get_all_images()

# Create columns for image selection
cols = st.columns(len(selectable_images))
selected_image = None

for idx, img_path in enumerate(selectable_images):
    with cols[idx]:
        img = Image.open(img_path)
        st.image(img, caption=img_path.stem, use_column_width=True)
        if st.button(f"Selecionar", key=f"btn_{idx}"):
            st.session_state['selected_image'] = str(img_path)

# Check if image is selected
if 'selected_image' in st.session_state:
    selected_path = Path(st.session_state['selected_image'])
    
    st.markdown("---")
    st.markdown("### <span class='step-badge'>2</span> Imagens Mais Similares", unsafe_allow_html=True)
    
    # Show selected image
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("**Imagem selecionada:**")
        st.image(Image.open(selected_path), caption=selected_path.stem, width=150)
    
    with col2:
        # Find similar images
        with st.spinner("Buscando imagens similares..."):
            similar_images = find_similar_images(extractor, selected_path, all_images, top_k=5)
        
        st.write("**Top 5 imagens mais similares:**")
        sim_cols = st.columns(5)
        for idx, result in enumerate(similar_images):
            with sim_cols[idx]:
                img = Image.open(result['path'])
                st.image(img, use_column_width=True)
                st.markdown(f"<div style='text-align:center'><span class='similarity-score'>{result['similarity']:.1%}</span></div>", unsafe_allow_html=True)
                st.caption(result['name'])

    # Step 3: Heatmaps
    st.markdown("---")
    st.markdown("### <span class='step-badge'>3</span> Mapas de Calor", unsafe_allow_html=True)
    st.write("Visualize quais regiões da imagem selecionada são similares às imagens encontradas.")
    
    # Generate heatmaps for top 3
    top_3 = similar_images[:3]
    heatmap_cols = st.columns(3)
    
    for idx, result in enumerate(top_3):
        with heatmap_cols[idx]:
            with st.spinner(f"Gerando heatmap {idx+1}..."):
                heatmap_img, overlay_img, cross_sim_mean = generate_heatmap(
                    extractor, heatmap_gen, selected_path, result['path']
                )
            
            st.write(f"**vs {result['name']}**")
            st.write(f"Similaridade: {result['similarity']:.1%}")
            
            # Toggle between heatmap and overlay
            view_mode = st.radio(
                "Visualização:",
                ["Overlay", "Heatmap"],
                key=f"view_{idx}",
                horizontal=True
            )
            
            if view_mode == "Overlay":
                st.image(overlay_img, use_column_width=True)
            else:
                st.image(heatmap_img, use_column_width=True)
            
            st.caption(f"Cross-similarity média: {cross_sim_mean:.1%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Powered by DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m)</p>
    <p>INPI - Instituto Nacional da Propriedade Industrial</p>
</div>
""", unsafe_allow_html=True)
