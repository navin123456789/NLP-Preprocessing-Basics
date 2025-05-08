from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging, asyncio
from utils.stem import Stemmer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/stemm_router.log')

try:
    GLOBAL_STEMMER_INSTANCE = Stemmer()
    logging.info("Stemmer initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Stemmer: {e}")
    GLOBAL_STEMMER_INSTANCE = None

def get_stemmer() -> Stemmer:
    if not GLOBAL_STEMMER_INSTANCE:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stemmer service unavailable.")
    return GLOBAL_STEMMER_INSTANCE

class TextInput(BaseModel):
    text: str

class StemmingResponse(BaseModel):
    stems: list[str]

router = APIRouter()

@router.post("/stemmize", response_model=StemmingResponse, status_code=status.HTTP_200_OK)
async def stemm_data(data: TextInput, stemmer: Stemmer = Depends(get_stemmer)):
    try:
        stems = await asyncio.to_thread(stemmer.stem, text=data.text)
        return StemmingResponse(stems=stems)
    except Exception as e:
        logging.error(f"Error during stemming: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}")