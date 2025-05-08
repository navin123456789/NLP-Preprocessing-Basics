import spacy, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='./logs/ner.log')

class NerRecognizer:
    """
    A Named Entity Recognition (NER) using spaCy.
    """
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes the NerRecognizer with a specified spaCy language model.
        
        Args:
            model_name (str, optional): The name of the spaCy model to load.
                                        Defaults to "en_core_web_sm" (English, small).
        
        Raises:
            OSError: If the specified spaCy model cannot be loaded.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            logging.error(f"Error loading spaCy model '{model_name}'. Install it via `python -m spacy download {model_name}`.")
            raise e

    def recognize_entities(self, text: str) -> list[tuple[str, str, tuple[int, int]]]:
        """
        Performs Named Entity Recognition (NER) on the input string using the loaded spaCy model.
        
        Args:
            text (str): The input string to be analyzed.
            
        Returns:
            list[tupe[str, str, tupel[int, int]]: A list of (entity text, entity type, (start char, end char)) tupels.
            Returns an empty list if the input is None or an empty string.
        
        Raises:
            TypeError: If the input is not a string.
        """
        if not isinstance(text, str) or not text.strip():
            logging.error(f"Invalid input: {type(text)}")
            return []
        return [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in self.nlp(text).ents]