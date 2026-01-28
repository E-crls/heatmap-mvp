import uvicorn
from api.retriever import app
from settings.vars_config import Config

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.API_PORT)