from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
import logging, asyncio
from utils.embedding import GloveEmbeddingGenerator
from app.utils.dimensions import DimensionReduction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/embedding_router.log')

try:
    GLOBAL_EMBEDDING_INSTANCE = GloveEmbeddingGenerator()
    logging.info("Embedding initialized successfully.")
except Exception as e:
    logging.critical(f"Failed to initialize Embedding: {e}")
    GLOBAL_EMBEDDING_INSTANCE = None

def get_embedding() -> GloveEmbeddingGenerator:
    if not GLOBAL_EMBEDDING_INSTANCE:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedder service unavailable.")
    return GLOBAL_EMBEDDING_INSTANCE

class TextInput(BaseModel):
    text: list[str]

class WordEmbeddings(BaseModel):
    word: str
    embedding: list

class ReducedResponse(BaseModel):
    reduced: list[WordEmbeddings]

router = APIRouter()

@router.post("/embed", response_model=ReducedResponse, status_code=status.HTTP_200_OK)
async def embed_data(data: TextInput, embedd: GloveEmbeddingGenerator = Depends(get_embedding)):
    try:
        embeddings = await asyncio.to_thread(embedd.get_word_embeddings, words=data.text)
        reducer = DimensionReduction(embeddings_dict=embeddings)
        reduced_data = reducer.perform_pca(n_components=2)
        response = reducer.get_reduced_data()
        return ReducedResponse(reduced=response or [])
    except Exception as e:
        logging.error(f"Error during embedding: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error: {e}")