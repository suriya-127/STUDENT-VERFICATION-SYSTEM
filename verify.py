try:
    from sentence_transformers import SentenceTransformer, util
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "sentence_transformers is not installed in the active Python environment. "
        "Run `.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt` "
        "and then start Streamlit with `.\\.venv\\Scripts\\python.exe -m streamlit run app.py`."
    ) from exc

from auva import AUVA
from concept_detector import ConceptDetector
from emotion_analysis import EmotionAnalyzer
from hesitation import HesitationAnalyzer
from misconception import MisconceptionDetector
from mediapipe_analysis import AttentionAnalyzer
from report import ReportGenerator


class SemanticAnalyzer:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def calculate_similarity(self, reference: str, transcript: str) -> float:
        if not reference or not transcript:
            return 0.0

        embeddings = self.model.encode([reference, transcript], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarity = max(min(similarity, 1.0), -1.0)
        return float(similarity)


def analyze_semantic(reference_text: str, transcript_text: str) -> dict:
    # Use the same sentence-level method as the pipeline for consistency
    from embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    sentences = engine.split_sentences(transcript_text)
    if not sentences and transcript_text:
        sentences = [transcript_text]

    sentence_embeddings = engine.encode_sentences(sentences, convert_to_tensor=True) if sentences else []

    ideal_sentences = engine.split_sentences(reference_text)
    ideal_embeddings = engine.encode_sentences(ideal_sentences, convert_to_tensor=True) if ideal_sentences else []

    sentence_scores = []
    for emb in sentence_embeddings:
        if len(ideal_embeddings) > 0:
            sims = util.cos_sim(emb, ideal_embeddings)
            best = float(sims.max().item())
        else:
            best = 0.0
        best = max(min(best, 1.0), -1.0)
        sentence_scores.append(max(0.0, best) * 100)

    avg = float(sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
    return {
        "sentence_scores": [round(s, 2) for s in sentence_scores],
        "average": round(avg, 2),
        "score": round(avg, 2),
        "reference": reference_text,
        "transcript": transcript_text,
    }
    


def analyze_concepts(concepts: list[str], transcript_text: str) -> dict:
    # Delegate to the pipeline's concept detection for consistency
    try:
        from pipeline import VerificationPipeline
    except Exception:
        detector = ConceptDetector()
        return detector.detect(concepts, transcript_text)

    sentences = []
    from embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    sentences = engine.split_sentences(transcript_text)
    if not sentences and transcript_text:
        sentences = [transcript_text]

    pipeline = VerificationPipeline()
    detected = pipeline.detect_concepts(concepts if isinstance(concepts, dict) else {c: 1 for c in concepts}, sentences)
    missing = {c: (concepts[c] if isinstance(concepts, dict) else 1) for c in (concepts.keys() if isinstance(concepts, dict) else concepts) if c not in detected}
    total = len(concepts) if isinstance(concepts, (list, dict)) else 0
    coverage = (sum(d.get("weight", 1) for d in detected.values()) / sum((concepts.values() if isinstance(concepts, dict) else [1]*total))) * 100 if total > 0 else 0.0

    performance = "Needs Improvement"
    if coverage >= 90:
        performance = "Excellent"
    elif coverage >= 75:
        performance = "Good"
    elif coverage >= 60:
        performance = "Average"

    return {
        "detected": list(detected.keys()),
        "missing": list(missing.keys()),
        "coverage": round(coverage, 2),
        "performance": performance,
    }


def analyze_misconceptions(
    transcript_text: str,
    misconceptions: list[str],
    ideal_answer: str,
) -> list[dict]:
    detector = MisconceptionDetector()
    # Split transcript into sentences and run detector per sentence
    from embedding_engine import EmbeddingEngine
    engine = EmbeddingEngine()
    sentences = engine.split_sentences(transcript_text)
    if not sentences and transcript_text:
        sentences = [transcript_text]

    results = []
    for sent in sentences:
        res = detector.detect(sent, misconceptions, ideal_answer)
        for item in res:
            item["sentence"] = sent
            results.append(item)

    return results


def analyze_attention(video_path: str) -> dict:
    analyzer = AttentionAnalyzer()
    return analyzer.analyze(video_path)


def analyze_emotion(video_path: str) -> dict:
    analyzer = EmotionAnalyzer()
    return analyzer.analyze(video_path)


def analyze_hesitation(timestamps: list[dict]) -> dict:
    analyzer = HesitationAnalyzer()
    return analyzer.analyze(timestamps)


def calculate_auva(
    semantic: dict,
    concept: dict,
    attention: dict,
    hesitation: dict,
    misconceptions: list[dict],
) -> dict:
    calculator = AUVA()
    return calculator.calculate(semantic, concept, attention, hesitation, misconceptions)


def analyze_auva(
    semantic: dict,
    concept: dict,
    attention: dict,
    hesitation: dict,
    misconceptions: list[dict],
) -> dict:
    return calculate_auva(semantic, concept, attention, hesitation, misconceptions)


def generate_report(
    transcript: str,
    semantic: dict,
    concept: dict,
    attention: dict,
    emotion: dict,
    hesitation: dict,
    misconception: list[dict],
    auva: dict,
) -> dict:
    generator = ReportGenerator()
    return generator.generate(
        transcript,
        semantic,
        concept,
        attention,
        emotion,
        hesitation,
        misconception,
        auva,
    )


def run_unified_verification(reference: dict, video_path: str, transcript: str, timestamps: list) -> dict:
    """Run the new unified VerificationPipeline and return its results.

    Returns a dict with keys: 'auva' and 'report'.
    """
    try:
        from pipeline import VerificationPipeline
    except Exception as exc:
        raise

    pipeline = VerificationPipeline()
    return pipeline.run(reference, video_path, transcript, timestamps)
