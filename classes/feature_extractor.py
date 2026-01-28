import torch
import base64
import io
from typing import Union, List
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
import sys
import os

# Adiciona o diretório raiz ao sys.path para permitir a importação de settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.vars_config import Config

class FeatureExtractor:
    """
    Classe para extração de embeddings de imagens usando modelos DINOv3 ou SigLIP2.
    Suporta entrada como imagem PIL ou string base64.
    """
    
    # Mapeamento de nomes curtos para IDs do Hugging Face
    # Nota: Usando versões atuais como fallback/exemplo caso as versões futuras (v3, 2) não estejam disponíveis no hub padrão
    MODEL_MAPPINGS = {
        "clip": "openai/clip-vit-large-patch14",
        "dino": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "siglip": "google/siglip2-so400m-patch14-224",
    }

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Inicializa o extrator de características.
        
        Args:
            model_name (str): Nome do modelo ('dinov3', 'siglip2') ou ID do Hugging Face.
            device (str): Dispositivo para execução ('cpu', 'cuda').
        """
        self.device = device
        self.model_name = model_name
        
        # Resolve o ID do modelo
        self.hf_model_id = self.MODEL_MAPPINGS.get(model_name.lower(), model_name)
        
        print(f"Carregando modelo: {self.hf_model_id} no dispositivo: {self.device}")
        
        try:
            token = Config.HF_TOKEN
            if "siglip" in self.hf_model_id.lower():
                self.processor = AutoProcessor.from_pretrained(self.hf_model_id, use_fast=True, token=token)
                self.model = AutoModel.from_pretrained(self.hf_model_id, token=token).to(self.device)
                self.model_type = "siglip"
            else:
                # Assume comportamento padrão (DINO e outros ViTs)
                self.processor = AutoImageProcessor.from_pretrained(self.hf_model_id, use_fast=True, token=token)
                self.model = AutoModel.from_pretrained(self.hf_model_id, token=token).to(self.device)
                self.model_type = "dino" # ou genérico
                
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar o modelo {self.hf_model_id}: {e}")

    def _process_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """Converte base64 para PIL Image se necessário e garante RGB."""
        if isinstance(image, str):
            try:
                # Remove cabeçalho de data URI se existir (ex: data:image/png;base64,...)
                if "base64," in image:
                    image = image.split("base64,")[1]
                
                image_data = base64.b64decode(image)
                img = Image.open(io.BytesIO(image_data))
            except Exception as e:
                raise ValueError(f"Falha ao decodificar imagem base64: {e}")
        else:
            img = image
            
        return img.convert("RGB")

    def to_embeddings(self, image: Union[str, Image.Image]) -> List[float]:
        """
        Gera embeddings para uma imagem.
        
        Args:
            image (Union[str, Image.Image]): Imagem PIL ou string base64.
            
        Returns:
            List[float]: Lista de floats representando o embedding.
        """
        img = self._process_image(image)
        
        try:
            # Preprocessamento
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.model_type == "siglip":
                    # SigLIP/CLIP usa get_image_features para projeção correta
                    if hasattr(self.model, "get_image_features"):
                        embeddings = self.model.get_image_features(**inputs)
                    else:
                        # Fallback
                        outputs = self.model(**inputs)
                        embeddings = outputs.pooler_output
                else:
                    # DINO e outros ViTs
                    outputs = self.model(**inputs)
                    # DINOv2 geralmente usa o CLS token (primeiro token)
                    # last_hidden_state shape: (batch, seq_len, hidden_size)
                    embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalização (L2 norm) - importante para busca vetorial (cosine similarity)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
            
            return embeddings.cpu().numpy().flatten().tolist()
            
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embeddings: {e}")
