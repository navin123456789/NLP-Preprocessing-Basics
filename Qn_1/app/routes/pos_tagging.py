from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging, asyncio
from utils.pos_tags import PosTagger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='logs/pos_router.log')

try:
    GLOBAL_POS_INSTANCE = PosTagger()
    logging.info("POS tagger initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize POS tagger: {e}")
    GLOBAL_POS_INSTANCE = None

def get_pos() -> PosTagger:
    if not GLOBAL_POS_INSTANCE:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="POS tagger service unavailable.")
    return GLOBAL_POS_INSTANCE

class TextInput(BaseModel):
    text: str

class PosResponse(BaseModel):
    pos: list[tuple[str, str]]

router = APIRouter()

@router.post("/pos", response_model=PosResponse, status_code=status.HTTP_200_OK)
async def recognize_data(data: TextInput, pos_tag: PosTagger = Depends(get_pos)):
    try:
        pos = await asyncio.to_thread(pos_tag.get_pos_tags, text=data.text)
        return PosResponse(pos=pos)
    except Exception as e:
        logging.error(f"Error during POS tagging: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}")