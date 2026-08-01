import cv2
from collections import Counter

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover - optional dependency fallback
    DeepFace = None


class EmotionAnalyzer:
    def __init__(self):
        pass

    def analyze(self, video_path: str) -> dict:
        if DeepFace is None:
            return {
                "status": "Unavailable",
                "reason": "DeepFace is not available in the current environment.",
                "confidence": 0.0,
                "dominant": None,
                "distribution": {},
                "analysis": "Emotion analysis is unavailable because DeepFace could not be imported.",
                "interpretation": "Emotion analysis is unavailable because DeepFace could not be imported.",
            }

        cap = cv2.VideoCapture(video_path)
        emotions = []
        frame_count = 0

        if not cap.isOpened():
            return {
                "dominant": None,
                "distribution": {},
                "analysis": "Video could not be opened for emotion analysis.",
                "interpretation": "Video could not be opened for emotion analysis.",
            }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:
                continue

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
            except Exception as e:
                print(f"DeepFace emotion error: {e}")
                continue

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            emotion = None
            if isinstance(result, dict):
                emotion = result.get("dominant_emotion")
                if not emotion:
                    emotion_scores = result.get("emotion")
                    if isinstance(emotion_scores, dict) and emotion_scores:
                        emotion = max(emotion_scores, key=emotion_scores.get)

            if emotion:
                emotions.append(emotion)

        cap.release()

        if not emotions:
            return {
                "status": "Unavailable",
                "reason": "Face was not consistently detected during the video.",
                "confidence": 0.0,
                "dominant": None,
                "distribution": {},
                "analysis": "Emotion analysis could not be completed because faces were not detected consistently.",
                "interpretation": "Emotion information is unavailable because faces were not detected consistently.",
            }

        emotion_count = Counter(emotions)
        total = len(emotions)
        summary = {
            emotion: round(count / total * 100, 2)
            for emotion, count in emotion_count.items()
        }
        dominant = emotion_count.most_common(1)[0][0]

        interpretation_lines = [
            f"Dominant Emotion: {dominant}",
            "Emotion Distribution:",
        ]
        for emotion, percent in summary.items():
            interpretation_lines.append(f"- {emotion}: {percent}%")

        if dominant.lower() in ["neutral", "happy", "surprise"]:
            interpretation_lines.append(
                "The student showed stable emotional engagement during the answer."
            )
        else:
            interpretation_lines.append(
                "Some emotional reactions were observed, which may reflect moments of uncertainty or difficulty."
            )
        interpretation_lines.append(
            "Emotion analysis is supporting evidence only and does not by itself determine topic understanding."
        )

        return {
            "dominant": dominant,
            "distribution": summary,
            "analysis": "\n".join(interpretation_lines),
            "interpretation": "\n".join(interpretation_lines),
        }
