from sentence_transformers import SentenceTransformer, util


class MisconceptionDetector:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _split_ideal_sentences(self, ideal_answer: str) -> list[str]:
        if not ideal_answer:
            return []

        return [
            sentence.strip()
            for sentence in ideal_answer.replace("?", ".").replace("!", ".").split(".")
            if sentence.strip()
        ]

    def detect(
        self,
        sentence: str,
        misconceptions: list[str],
        ideal_sentences: list[str] | str | None = None,
    ) -> list[dict]:
        results = []

        if not sentence or not misconceptions:
            return results

        if isinstance(ideal_sentences, str):
            ideal_sentences = self._split_ideal_sentences(ideal_sentences)

        ideal_sentences = ideal_sentences or []
        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
        ideal_embeddings = (
            self.model.encode(ideal_sentences, convert_to_tensor=True)
            if ideal_sentences
            else None
        )

        for item in misconceptions:
            if not item:
                continue

            mis_embedding = self.model.encode(item, convert_to_tensor=True)
            mis_similarity = float(util.cos_sim(sentence_embedding, mis_embedding).item())

            ideal_similarity = 0.0
            closest_ideal = ""
            if ideal_embeddings is not None and len(ideal_sentences) > 0:
                sims = util.cos_sim(sentence_embedding, ideal_embeddings)
                best_idx = int(sims.argmax().item())
                ideal_similarity = float(sims[0, best_idx].item())
                closest_ideal = ideal_sentences[best_idx]

            margin = 0.10
            if mis_similarity > 0.70 and mis_similarity > ideal_similarity + margin:
                reason = f"""
The student's sentence matched a common misconception more strongly than the closest correct statement.

Sentence: \"{sentence}\"
Misconception: \"{item}\"
Correct concept: \"{closest_ideal}\"
"""
                results.append({
                    "student_sentence": sentence,
                    "misconception": item,
                    "correct_concept": closest_ideal,
                    "mis_similarity": round(mis_similarity * 100, 2),
                    "ideal_similarity": round(ideal_similarity * 100, 2),
                    "reason": reason.strip(),
                })

        return results
