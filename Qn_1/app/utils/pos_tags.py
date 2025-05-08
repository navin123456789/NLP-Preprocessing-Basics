import spacy, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/pos_tag.log')

class PosTagger:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            logging.error(f"Error loading spaCy model '{model_name}'. Install it via `python -m spacy download {model_name}`.")
            raise e

    def get_pos_tags(self, text: str) -> list[tuple[str, str]]:
        if not isinstance(text, str) or not text.strip():
            logging.error(f"Invalid input: {type(text)}")
            return []
        return [(token.text, token.tag_) for token in self.nlp(text)]