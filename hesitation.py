import statistics


class HesitationAnalyzer:
    def __init__(self):
        pass

    def analyze(self, timestamps: list[dict]) -> dict:
        if not timestamps or len(timestamps) < 2:
            return {
                "total_pauses": 0,
                "long_pauses": 0,
                "average_pause": 0.0,
                "maximum_pause": 0.0,
                "words_per_minute": 0.0,
                "confidence": 0.0,
                "performance": "Needs More Practice",
            }

        pauses = []
        long_pauses = []
        total_words = len(timestamps)

        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1]["start"] - timestamps[i]["end"]
            if gap > 0:
                pauses.append(gap)
                if gap >= 2:
                    long_pauses.append(gap)

        average_pause = statistics.mean(pauses) if pauses else 0.0
        maximum_pause = max(pauses) if pauses else 0.0
        duration = timestamps[-1]["end"] - timestamps[0]["start"]
        wpm = (total_words / duration) * 60 if duration > 0 else 0.0

        pause_penalty = average_pause * 8
        long_pause_penalty = len(long_pauses) * 4
        # Prefer speaking rate in the 120-160 WPM band. Penalize outside this range.
        if 120 <= wpm <= 160:
            speed_penalty = 0.0
        elif wpm < 120:
            speed_penalty = (120 - wpm) * 0.3
        else:
            speed_penalty = (wpm - 160) * 0.4
        confidence = 100 - pause_penalty - long_pause_penalty - speed_penalty
        confidence = max(0.0, min(100.0, confidence))

        if confidence >= 90:
            level = "Highly Confident"
        elif confidence >= 75:
            level = "Confident"
        elif confidence >= 60:
            level = "Moderately Confident"
        else:
            level = "Needs More Practice"

        return {
            "total_pauses": len(pauses),
            "long_pauses": len(long_pauses),
            "average_pause": round(average_pause, 2),
            "maximum_pause": round(maximum_pause, 2),
            "words_per_minute": round(wpm, 2),
            "confidence": confidence,
            "performance": level,
        }
