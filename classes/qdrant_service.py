import os
import sys
from typing import List, Dict, Any, Optional, Union
from qdrant_client import QdrantClient, models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.vars_config import Config

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
            api_key=Config.QDRANT_API_KEY
        )

    def get_point(self, collection_name: str, point_id: Union[int, str]) -> Dict[str, Any]:
        """
        Recupera um ponto pelo ID.
        
        Args:
            collection_name (str): Nome da coleção.
            point_id (Union[int, str]): ID do ponto.

        Returns:
            Dict[str, Any]: O payload do ponto ou erro.
        """
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            if points:
                return {"status": "success", "point": points[0].payload}
            else:
                return {"status": "error", "message": "Point not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search(self, collection_name: str, embedding: List[float], limit: int = 5, query_filter: Optional[models.Filter] = None) -> Dict[str, Any]:
        """
        Realiza uma busca no Qdrant usando um vetor de embedding.
        
        Args:
            collection_name (str): Nome da coleção no Qdrant.
            embedding (List[float]): O vetor de embedding para busca.
            limit (int): Número máximo de resultados.
            query_filter (Optional[models.Filter]): Filtro para a busca.

        Returns:
            Dict[str, Any]: Um dicionário contendo a lista de resultados ou erro.
        """
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                query_filter=query_filter,
                limit=limit
            )
            
            # Formata o resultado como uma lista de dicionários
            formatted_results = []
            for point in search_result:
                formatted_results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
                
            return {"status": "success", "results": formatted_results}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Verifica a conexão com o Qdrant.
        """
        try:
            collections = self.client.get_collections()
            return {"status": "success", "message": "Connected to Qdrant", "collections_count": len(collections.collections)}
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to Qdrant: {str(e)}"}