import re

from sentence_transformers import SentenceTransformer, util


class ConceptDetector:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _split_sentences(self, transcript: str) -> list[str]:
        transcript = transcript.strip()
        if not transcript:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", transcript)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def detect(self, concepts: list[str], transcript: str) -> dict:
        detected = []
        missing = []

        if not transcript:
            missing = list(concepts or [])
            total = len(concepts or [])
            coverage = 0.0
            performance = "Needs Improvement"
            return {
                "detected": detected,
                "missing": missing,
                "coverage": coverage,
                "performance": performance,
            }

        sentences = self._split_sentences(transcript)
        transcript_inputs = sentences if sentences else [transcript]
        transcript_embeddings = self.model.encode(transcript_inputs, convert_to_tensor=True)

        for concept in concepts or []:
            if not concept:
                missing.append(concept)
                continue

            concept_embedding = self.model.encode(concept, convert_to_tensor=True)
            similarity_scores = util.cos_sim(concept_embedding, transcript_embeddings)
            score = float(similarity_scores.max().item())

            if score >= 0.55:
                detected.append(concept)
            else:
                missing.append(concept)

        total = len(concepts or [])
        coverage = (len(detected) / total) * 100.0 if total > 0 else 0.0

        if coverage >= 90:
            performance = "Excellent"
        elif coverage >= 75:
            performance = "Good"
        elif coverage >= 60:
            performance = "Average"
        else:
            performance = "Needs Improvement"

        return {
            "detected": detected,
            "missing": missing,
            "coverage": coverage,
            "performance": performance,
        }
