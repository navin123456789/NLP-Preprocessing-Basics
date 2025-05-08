from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging, asyncio
from utils.ner import NerRecognizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/ner_router.log')

try:
    GLOBAL_NER_INSTANCE = NerRecognizer()
    logging.info("NER initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize NER: {e}")
    GLOBAL_NER_INSTANCE = None

def get_ner() -> NerRecognizer:
    if not GLOBAL_NER_INSTANCE:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="NER service unavailable.")
    return GLOBAL_NER_INSTANCE

class TextInput(BaseModel):
    text: str

class NerResponse(BaseModel):
    ner: list[tuple[str, str, tuple[int, int]]]

router = APIRouter()

@router.post("/ner", response_model=NerResponse, status_code=status.HTTP_200_OK)
async def recognize_data(data: TextInput, ner: NerRecognizer = Depends(get_ner)):
    try:
        ner_result = await asyncio.to_thread(ner.recognize_entities, text=data.text)
        return NerResponse(ner=ner_result)
    except Exception as e:
        logging.error(f"Error during NER: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}")