import cv2

try:
    import mediapipe as mp
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    mp = None
    np = None


class AttentionAnalyzer:
    def __init__(self):
        self.face_mesh = None
        if mp is not None and hasattr(mp, "solutions"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def _landmark_point(self, landmark, frame_width, frame_height):
        return np.array([landmark.x * frame_width, landmark.y * frame_height])

    def _calculate_head_angle(self, landmarks, frame_width, frame_height):
        nose = self._landmark_point(landmarks[1], frame_width, frame_height)
        chin = self._landmark_point(landmarks[199], frame_width, frame_height)
        vector = chin - nose
        angle = np.degrees(np.arctan2(vector[0], vector[1]))
        return abs(angle)

    def _is_looking_center(self, landmarks, frame_width, frame_height):
        left_eye = self._landmark_point(landmarks[33], frame_width, frame_height)
        left_iris = self._landmark_point(landmarks[468], frame_width, frame_height)
        right_eye = self._landmark_point(landmarks[263], frame_width, frame_height)
        right_iris = self._landmark_point(landmarks[473], frame_width, frame_height)

        left_offset = abs(left_iris[0] - left_eye[0])
        right_offset = abs(right_iris[0] - right_eye[0])
        threshold = frame_width * 0.03

        return left_offset < threshold and right_offset < threshold

    def analyze(self, video_path: str) -> dict:
        if self.face_mesh is None or np is None:
            return {
                "attention_score": 0.0,
                "face_visibility": 0.0,
                "head_score": 0.0,
                "eye_score": 0.0,
                "stability": 0.0,
                "performance": "Unavailable",
            }

        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        face_frames = 0
        looking_center = 0
        looking_away = 0
        nose_positions = []

        if not cap.isOpened():
            return {
                "attention_score": 0.0,
                "face_visibility": 0.0,
                "head_score": 0.0,
                "eye_score": 0.0,
                "stability": 0.0,
                "performance": "Poor",
            }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            frame_height, frame_width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if result.multi_face_landmarks:
                face_frames += 1
                landmarks = result.multi_face_landmarks[0].landmark

                angle = self._calculate_head_angle(landmarks, frame_width, frame_height)
                if angle < 15:
                    pass
                else:
                    looking_away += 1

                if self._is_looking_center(landmarks, frame_width, frame_height):
                    looking_center += 1
                else:
                    looking_away += 1

                nose = self._landmark_point(landmarks[1], frame_width, frame_height)
                nose_positions.append(nose)

        cap.release()

        visibility = (face_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
        eye_score = (looking_center / face_frames) * 100.0 if face_frames > 0 else 0.0
        head_score = max(0.0, 100.0 - (looking_away / face_frames) * 100.0) if face_frames > 0 else 0.0

        stability = 0.0
        if len(nose_positions) > 1:
            movements = [
                np.linalg.norm(nose_positions[i] - nose_positions[i - 1])
                for i in range(1, len(nose_positions))
            ]
            avg_movement = float(np.mean(movements)) / np.linalg.norm([frame_width, frame_height])
            stability = max(0.0, 100.0 - min(100.0, avg_movement * 120.0))

        attention = (
            0.40 * visibility
            + 0.30 * head_score
            + 0.20 * eye_score
            + 0.10 * stability
        )

        if attention >= 90:
            level = "Excellent"
        elif attention >= 75:
            level = "Good"
        elif attention >= 60:
            level = "Average"
        else:
            level = "Poor"

        return {
            "attention_score": round(attention, 2),
            "face_visibility": round(visibility, 2),
            "head_score": round(head_score, 2),
            "eye_score": round(eye_score, 2),
            "stability": round(stability, 2),
            "performance": level,
        }
