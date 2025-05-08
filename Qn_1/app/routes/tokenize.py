from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging, asyncio
from utils.token import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/token_router.log')

try:
    GLOBAL_TOKENIZER_INSTANCE = Tokenizer()
    logging.info("Tokenizer initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Tokenizer: {e}")
    GLOBAL_TOKENIZER_INSTANCE = None

def get_tokenizer() -> Tokenizer:
    if not GLOBAL_TOKENIZER_INSTANCE:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Tokenizer service unavailable.")
    return GLOBAL_TOKENIZER_INSTANCE

class TextInput(BaseModel):
    text: str

class TokenizeResponse(BaseModel):
    tokens: list[str]

router = APIRouter()

@router.post("/tokenize", response_model=TokenizeResponse, status_code=status.HTTP_200_OK)
async def tokenize_data(data: TextInput, tokenizer: Tokenizer = Depends(get_tokenizer)):
    try:
        tokens = await asyncio.to_thread(tokenizer.tokenize, text=data.text)
        return TokenizeResponse(tokens=tokens)
    except Exception as e:
        logging.error(f"Error during tokenization: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}")