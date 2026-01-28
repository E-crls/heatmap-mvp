from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, Union
import json
import base64
from PIL import Image
import io
import os
from settings.vars_config import Config
from settings.inpi_config import hybrid_search
from classes.qdrant_service import QdrantService
from classes.feature_extractor import FeatureExtractor
from classes.inpi_interface import InpiInterface

app = FastAPI()

# Mount static files
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/health")
async def health_check():
    qdrant_health = qdrant.health_check()
    status = "healthy" if qdrant_health["status"] == "success" else "unhealthy"
    return {
        "status": status,
        "qdrant": qdrant_health,
        "api": "online"
    }

@app.get("/brand")
async def get_brand(id: str, collection_name: str = Config.SIGLIP_COLLECTION):
    # Tenta converter para int se possível, já que IDs numéricos são comuns
    try:
        point_id = int(id)
    except ValueError:
        point_id = id

    result = qdrant.get_point(collection_name=collection_name, point_id=point_id)
    
    if result["status"] == "success":
        return result["point"]
    else:
        return result

qdrant = QdrantService()

siglip_extractor = FeatureExtractor(model_name="siglip")
siglip_interface = InpiInterface(feature_extractor=siglip_extractor, qdrant_service=qdrant)

dino_extractor = FeatureExtractor(model_name="dino")
dino_interface = InpiInterface(feature_extractor=dino_extractor, qdrant_service=qdrant)

@app.post("/search")
async def search(
    file: Union[UploadFile, str, None] = File(None), 
    image_base64: Optional[str] = Form(None),
    filters: Optional[str] = Form(None),
    top_k: int = Form(Config.TOP_K),
    model: str = Form("dino")
):
    image = None
    debug_info = {}

    if file and not isinstance(file, str):
        try:
            content = await file.read()
            debug_info["file_content_size"] = len(content)
            if content:
                image = Image.open(io.BytesIO(content))
        except Exception as e:
            debug_info["file_error"] = str(e)
    
    if image is None and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
             return {"status": "error", "message": f"Invalid base64 image: {str(e)}"}
    
    if image is None:
        return {
            "status": "error", 
            "message": "Either file or image_base64 must be provided",
            "received": {
                "file_type": str(type(file)),
                "image_base64_present": bool(image_base64),
                "debug_info": debug_info
            }
        }
    
    filters_dict = None
    if filters:
        try:
            filters_dict = json.loads(filters)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for filters"}
    
    if model == "hybrid":
        # Busca híbrida
        results_siglip = siglip_interface.search(
            image=image, 
            collection_name=Config.SIGLIP_COLLECTION, 
            top_k=top_k, 
            filters=filters_dict
        )
        results_dino = dino_interface.search(
            image=image, 
            collection_name=Config.DINOV3_COLLECTION, 
            top_k=top_k, 
            filters=filters_dict
        )
        
        combined_results = {}
        
        # Processa SigLIP
        if results_siglip.get("status") == "success":
            for item in results_siglip.get("results", []):
                item_id = item["id"]
                score = item["score"] * hybrid_search.get("siglip", 0.5)
                combined_results[item_id] = {
                    "id": item_id,
                    "score": score,
                    "payload": item["payload"],
                    "sources": ["siglip"]
                }
                
        # Processa DINO
        if results_dino.get("status") == "success":
            for item in results_dino.get("results", []):
                item_id = item["id"]
                score = item["score"] * hybrid_search.get("dino", 0.5)
                
                if item_id in combined_results:
                    combined_results[item_id]["score"] += score
                    combined_results[item_id]["sources"].append("dino")
                else:
                    combined_results[item_id] = {
                        "id": item_id,
                        "score": score,
                        "payload": item["payload"],
                        "sources": ["dino"]
                    }
        
        # Ordena e limita
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        resultado = final_results[:top_k]
        
        return resultado

    elif model == "dino":
        interface = dino_interface
        collection = Config.DINOV3_COLLECTION
    else:
        interface = siglip_interface
        collection = Config.SIGLIP_COLLECTION

    resultado = interface.search(
        image=image, 
        collection_name=collection, 
        top_k=top_k,
        filters=filters_dict
    )
    
    return resultado
