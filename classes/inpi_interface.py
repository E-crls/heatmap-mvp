from typing import Union, Dict, Any, Optional
from PIL import Image
import sys
import os
from qdrant_client import models

# Adiciona o diretório raiz ao sys.path para permitir a importação de settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings.inpi_config import fields_to_keep
from .qdrant_service import QdrantService
from .feature_extractor import FeatureExtractor

class InpiInterface:
    def __init__(self, qdrant_service: QdrantService, feature_extractor: FeatureExtractor):
        """
        Inicializa a interface do INPI.

        Args:
            qdrant_service (QdrantService): Instância do serviço Qdrant.
            feature_extractor (FeatureExtractor): Instância do extrator de características.
        """
        self.qdrant_service = qdrant_service
        self.feature_extractor = feature_extractor

    def search(self, image: Union[str, Image.Image], collection_name: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Realiza uma busca visual: gera embedding da imagem e consulta o Qdrant.

        Args:
            image (Union[str, Image.Image]): Imagem de entrada (caminho, base64 ou PIL Image).
            collection_name (str): Nome da coleção no Qdrant.
            limit (int): Número máximo de resultados.
            filters (Optional[Dict[str, Any]]): Filtros opcionais para a busca.

        Returns:
            Dict[str, Any]: Resultado da busca contendo os itens encontrados.
        """
        try:
            # 1. Gerar embedding
            embedding = self.feature_extractor.to_embeddings(image)

            # 2. Construir filtro Qdrant
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key in fields_to_keep:
                        if isinstance(value, list):
                            conditions.append(
                                models.FieldCondition(key=key, match=models.MatchAny(any=value))
                            )
                        else:
                            conditions.append(
                                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                            )
                if conditions:
                    query_filter = models.Filter(must=conditions)

            # 3. Buscar no Qdrant
            search_result = self.qdrant_service.search(
                collection_name=collection_name, 
                embedding=embedding, 
                limit=top_k,
                query_filter=query_filter
            )

            # Filtrar campos do payload
            if search_result.get("status") == "success":
                for item in search_result.get("results", []):
                    if "payload" in item and isinstance(item["payload"], dict):
                        item["payload"] = {
                            k: v for k, v in item["payload"].items() if k in fields_to_keep
                        }

            return search_result

        except Exception as e:
            return {
                "status": "error", 
                "message": f"Erro na busca INPI: {str(e)}"
            }
