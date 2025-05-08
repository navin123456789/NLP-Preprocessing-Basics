import nltk, logging
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/steamming.log')

class Stemmer:
    def __init__(self, stemmer_type='porter'):
        if stemmer_type not in ["porter", "snowball"]:
            logging.error(f"Invalid stemmer type: {stemmer_type}")
            raise ValueError("Must be 'porter' or 'snowball'")
        self.stemmer = PorterStemmer() if stemmer_type == 'porter' else SnowballStemmer("english")

    def stem(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            logging.error(f"Invalid input: {type(text)}")
            return []
        return [self.stemmer.stem(word) for word in word_tokenize(text)]