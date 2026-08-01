from sentence_transformers import SentenceTransformer
import re
from typing import List


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def split_sentences(self, text: str) -> List[str]:
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def encode_sentences(self, sentences: List[str], convert_to_tensor: bool = True):
        if not sentences:
            return []
        return self.model.encode(sentences, convert_to_tensor=convert_to_tensor)

    def encode_sentence(self, sentence: str, convert_to_tensor: bool = True):
        if not sentence:
            return self.model.encode("", convert_to_tensor=convert_to_tensor)
        return self.model.encode(sentence, convert_to_tensor=convert_to_tensor)

    def encode_text(self, text: str, convert_to_tensor: bool = True):
        if text is None:
            text = ""
        sentences = self.split_sentences(text)
        if sentences:
            return self.encode_sentences(sentences, convert_to_tensor=convert_to_tensor)
        return self.model.encode(text, convert_to_tensor=convert_to_tensor)
