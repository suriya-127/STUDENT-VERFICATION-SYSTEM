from sentence_transformers import util
from embedding_engine import EmbeddingEngine
from mediapipe_analysis import AttentionAnalyzer
from emotion_analysis import EmotionAnalyzer
from hesitation import HesitationAnalyzer
from misconception import MisconceptionDetector
from report import ReportGenerator
from typing import List


class VerificationPipeline:
    def __init__(self, model_name: str = "all-mpnet-base-v2", concept_threshold: float = 0.5):
        self.engine = EmbeddingEngine(model_name)
        self.model = self.engine.model
        self.concept_threshold = concept_threshold

    def detect_concepts(self, concept_weights: dict, sentences: list[str]) -> dict:
        """Detect concepts from a list of sentences. Returns a dict mapping concept->info."""
        detected = {}
        sentence_embeddings = self.engine.encode_sentences(sentences, convert_to_tensor=True) if sentences else []

        for concept, weight in concept_weights.items():
            aliases = []
            if isinstance(weight, dict):
                aliases = weight.get("aliases", []) or []
                actual_weight = weight.get("weight", 1)
            else:
                actual_weight = weight

            variants = [concept] + [a for a in aliases if a]

            best_sim = 0.0
            best_idx = -1
            best_variant = ""

            for variant in variants:
                v_emb = self.model.encode(variant, convert_to_tensor=True)
                if len(sentence_embeddings) > 0:
                    sims = util.cos_sim(v_emb, sentence_embeddings)
                    sim_val = float(sims.max().item())
                    sim_idx = int(sims.argmax().item())
                else:
                    sim_val = 0.0
                    sim_idx = -1

                if sim_val > best_sim:
                    best_sim = sim_val
                    best_idx = sim_idx
                    best_variant = variant

            if best_sim < self.concept_threshold:
                for i, sent in enumerate(sentences):
                    sent_l = sent.lower()
                    if concept.lower() in sent_l:
                        best_sim = max(best_sim, 0.6)
                        best_idx = i
                        best_variant = concept
                        break
                    for token in concept.split():
                        if len(token) > 3 and token.lower() in sent_l:
                            best_sim = max(best_sim, 0.58)
                            best_idx = i
                            best_variant = token
                            break
                    if best_sim >= self.concept_threshold:
                        break

            if best_sim >= self.concept_threshold:
                detected[concept] = {
                    "weight": actual_weight,
                    "similarity": round(best_sim * 100, 2),
                    "sentence_index": best_idx,
                    "sentence": sentences[best_idx] if 0 <= best_idx < len(sentences) else "",
                    "matched_variant": best_variant,
                }

        # heuristic inference
        if "Binary Search" in detected:
            if "Sorted Array" in concept_weights and "Sorted Array" not in detected:
                detected["Sorted Array"] = {
                    "weight": concept_weights.get("Sorted Array", 1) if not isinstance(concept_weights.get("Sorted Array"), dict) else concept_weights.get("Sorted Array", {}).get("weight", 1),
                    "similarity": round(0.75 * 100, 2),
                    "sentence_index": detected["Binary Search"].get("sentence_index", -1),
                    "sentence": detected["Binary Search"].get("sentence", ""),
                    "matched_variant": "inferred_from_Binary_Search",
                }

        return detected

    def run(self, reference: dict, video_path: str, transcript: str, timestamps: List[dict]) -> dict:
        sentences = self.engine.split_sentences(transcript)
        if not sentences and transcript:
            sentences = [transcript]

        sentence_embeddings = (
            self.engine.encode_sentences(sentences, convert_to_tensor=True)
            if sentences
            else []
        )

        ideal_text = reference.get("ideal_answer", "")
        ideal_sentences = self.engine.split_sentences(ideal_text)
        ideal_embeddings = (
            self.engine.encode_sentences(ideal_sentences, convert_to_tensor=True)
            if ideal_sentences
            else []
        )

        semantic_sentence_scores = []
        for emb in sentence_embeddings:
            if len(ideal_embeddings) > 0:
                sims = util.cos_sim(emb, ideal_embeddings)
                best = float(sims.max().item())
            else:
                best = 0.0
            best = max(min(best, 1.0), -1.0)
            semantic_sentence_scores.append(max(0.0, best) * 100)

        semantic_avg = float(sum(semantic_sentence_scores) / len(semantic_sentence_scores)) if semantic_sentence_scores else 0.0

        concepts = reference.get("concepts", {})
        if isinstance(concepts, list):
            concept_weights = {c: 1 for c in concepts}
        elif isinstance(concepts, dict):
            concept_weights = concepts
        else:
            concept_weights = {}

        total_weight = sum(concept_weights.values()) if concept_weights else 0
        # Use shared detection method
        detected = self.detect_concepts(concept_weights, sentences)

        # Heuristic inference: if Binary Search detected, infer Sorted Array if present
        if "Binary Search" in detected:
            if "Sorted Array" in concept_weights and "Sorted Array" not in detected:
                detected["Sorted Array"] = {
                    "weight": concept_weights.get("Sorted Array", 1) if not isinstance(concept_weights.get("Sorted Array"), dict) else concept_weights.get("Sorted Array", {}).get("weight", 1),
                    "similarity": round(0.75 * 100, 2),
                    "sentence_index": detected["Binary Search"].get("sentence_index", -1),
                    "sentence": detected["Binary Search"].get("sentence", ""),
                    "matched_variant": "inferred_from_Binary_Search",
                }

        detected_weight = sum(item["weight"] for item in detected.values())
        coverage = (detected_weight / total_weight) * 100 if total_weight > 0 else 0.0
        missing = {c: w for c, w in concept_weights.items() if c not in detected}
        missing_weight = sum(missing.values()) if missing else 0
        # Calculate raw knowledge gap as a percent of total concept weight
        gap_percent = (missing_weight / total_weight) * 100 if total_weight > 0 else 0.0
        # Scale the penalty so it cannot dominate the AUVA score (max penalty ~25)
        knowledge_gap_penalty = gap_percent * 0.25

        misconception_list = reference.get("misconceptions", [])
        mis_detector = MisconceptionDetector()
        misconceptions_found = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            res = mis_detector.detect(sentence, misconception_list, ideal_sentences)
            for item in res:
                item["sentence_index"] = i
                item["sentence"] = sentence
                misconceptions_found.append(item)

        attention = AttentionAnalyzer().analyze(video_path) if video_path else {
            "attention_score": 0.0,
            "face_visibility": 0.0,
            "head_score": 0.0,
            "eye_score": 0.0,
            "stability": 0.0,
            "performance": "Unavailable",
        }
        emotion = EmotionAnalyzer().analyze(video_path) if video_path else {
            "status": "Unavailable",
            "reason": "Video was not provided for emotion analysis.",
            "confidence": 0.0,
            "dominant": None,
            "distribution": {},
            "interpretation": "Emotion analysis is unavailable.",
        }
        hesitation = HesitationAnalyzer().analyze(timestamps) if timestamps else {
            "total_pauses": 0,
            "long_pauses": 0,
            "average_pause": 0.0,
            "maximum_pause": 0.0,
            "words_per_minute": 0.0,
            "confidence": 0.0,
            "performance": "Needs Practice",
        }

        semantic_score = semantic_avg
        concept_score = coverage
        attention_score = attention.get("attention_score", 0.0)
        hesitation_confidence = hesitation.get("confidence", 0.0)
        misconception_penalty = min(len(misconceptions_found) * 10, 50)

        if semantic_score < 40 and coverage < 40 and not misconceptions_found:
            misconception_note = "Answer is too unrelated for reliable misconception detection."
        else:
            misconception_note = ""

        understanding = (
            0.35 * semantic_score
            + 0.25 * concept_score
            + 0.15 * attention_score
            + 0.10 * hesitation_confidence
            - knowledge_gap_penalty
            - misconception_penalty
        )
        understanding = max(0.0, min(100.0, understanding))

        if understanding >= 85:
            performance = "Excellent"
        elif understanding >= 70:
            performance = "Good"
        elif understanding >= 50:
            performance = "Fair"
        else:
            performance = "Needs Improvement"

        auva = {
            "understanding_score": round(understanding, 2),
            "performance": performance,
            "semantic": {
                "sentence_scores": [round(s, 2) for s in semantic_sentence_scores],
                "average": round(semantic_avg, 2),
            },
            "concept": {
                "detected": detected,
                "missing": missing,
                "coverage": round(coverage, 2),
                "total_weight": total_weight,
            },
            "knowledge_gap_penalty": round(knowledge_gap_penalty, 2),
            "knowledge_gap_percent": round(gap_percent, 2),
            "misconceptions": misconceptions_found,
            "misconception_note": misconception_note,
            "attention": attention,
            "emotion": emotion,
            "hesitation": hesitation,
            "reasons": self._generate_reasons(
                semantic_sentence_scores,
                detected,
                missing,
                misconceptions_found,
                misconception_note,
            ),
        }

        report = ReportGenerator().generate(
            transcript,
            {"score": semantic_score},
            {
                "detected": list(detected.keys()),
                "missing": list(missing.keys()),
                "coverage": round(coverage, 2),
                "detected_count": len(detected),
                "total": len(concept_weights),
            },
            attention,
            emotion,
            hesitation,
            misconceptions_found,
            auva,
        )

        return {"auva": auva, "report": report}

    def _generate_reasons(self, sentence_scores, detected, missing, misconceptions_found, misconception_note):
        reasons = []
        low_indices = [i for i, s in enumerate(sentence_scores) if s < 40]
        for i in low_indices:
            reasons.append(f"Sentence {i+1} shows low semantic alignment ({sentence_scores[i]:.1f}%).")

        for c, w in missing.items():
            reasons.append(f"Missing concept: {c} (weight={w}).")

        for m in misconceptions_found:
            similarity = m.get('mis_similarity') or m.get('similarity')
            reasons.append(
                f"Misconception detected: {m.get('misconception')} (similarity={similarity}%)."
            )

        if misconception_note:
            reasons.append(misconception_note)

        if not reasons:
            reasons.append(
                "No major issues detected; student response appears consistent with the reference."
            )

        return reasons
