import numpy as np, logging
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/dimension.log')

class DimensionReduction:
    """
    A code to handle word embeddings, perform PCA dimensionality redutction,
    and export the results.
    """
    
    def __init__(self, embeddings_dict=None):
        """
        Initializes the DimensionReduction with a dictioniary of words and embeddings.
        
        Args:
            embeddings_dict (dict, optional): A dictionary where keys are words and values
                                                and values are lists or numpy arrays 
                                                representing their embeddings. Defaults to None.
        """
        self.embeddings_dict = embeddings_dict or {}
        self.words, self.embeddings = zip(*self.embeddings_dict.items()) if self.embeddings_dict else ([], None)

    def perform_pca(self, n_components=2):
        """
        Performs Principal Component Analysis (PCA) on the embeddings.
        
        Args:
            n_components (int): The number of components to keep after PCA.
                                Defaluts to 2 (for 2D visualization).

        Returns:
            numpy.ndarray or None: The reduced embeddings if successful, None otherwise.
        """
        if not self.embeddings:
            logging.error("No embeddings for PCA.")
            return None
        return PCA(n_components=n_components).fit_transform(np.array(self.embeddings))

    def get_reduced_data(self, reduced_embeddings):
        """
        Reduces the words and their corresponding embeddings.
        
        Retruns:
            list of dict or None: A list of dictionaries, where each dictionary
                                    contains 'word' and 'embeddings' (reduced) keys,
                                    or None if PCA has not been performed or failed.
        """
        return [{'word': word, 'embedding': emb.tolist()} for word, emb in zip(self.words, reduced_embeddings)]