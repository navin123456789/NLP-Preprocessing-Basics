import numpy as np
import os
import gensim.downloader as api, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/embedding.log')

class GloveEmbeddingGenerator:
    """
    A class to load a GloVe model and generate word embeddings.
    """
    
    def __init__(self, model_name='glove-wiki-gigaword-100'):
        """
        Initializes the specified model.

        Args:
            model_name: The name of the GloVe model to load. Defaults to 'glove-wiki-gigaword-100'.
        """
        self.model_name = model_name
        self._model = api.load(model_name) if model_name in api.info()['models'] else None

    def get_word_embedding(self, word: str):
        """
        Gets the vector embedding for a given word from the loaded GloVe model.
        
        Args:
            word: The word for which to get the embedding.
            
        Returns:
            A numpy array representing the word embedding, or None if the word is 
            not in the vocabulary or the model is not loaded.
        """
        if not self._model:
            logging.error("Model not loaded.")
            return None
        return self._model.get(word, None)

    def get_word_embeddings(self, words: list[str]):
        """
        Gets vector embeddings for a list of words.
        
        Args:
            words: A list of words for which to get embeddings.

        Returns:
            A dictionary where keys are words and values are their numpy array embeddings.
            Words not found in the vocabulary will not be included in the dictionary.
        """
        return {word: self.get_word_embedding(word) for word in words if self.get_word_embedding(word)}