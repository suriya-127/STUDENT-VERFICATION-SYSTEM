class AUVA:
    def __init__(self):
        pass

    def calculate(
        self,
        semantic: dict,
        concept: dict,
        attention: dict,
        hesitation: dict,
        misconceptions: list[dict],
    ) -> dict:
        semantic_score = semantic.get("score", 0.0)
        concept_score = concept.get("coverage", 0.0)
        attention_score = attention.get("attention_score", 0.0)
        confidence_score = hesitation.get("confidence", 0.0)
        mis_count = len(misconceptions or [])
        knowledge_gap = len(concept.get("missing", []))

        score = (
            semantic_score * 0.35
            + concept_score * 0.30
            + attention_score * 0.15
            + confidence_score * 0.10
        )

        score -= knowledge_gap * 2
        score -= mis_count * 3
        score = max(0.0, min(score, 100.0))

        if score >= 90:
            level = "Excellent"
        elif score >= 75:
            level = "Good"
        elif score >= 60:
            level = "Average"
        else:
            level = "Needs Improvement"

        reasons = []
        if semantic_score < 70:
            reasons.append(
                "The semantic similarity between the student's explanation and the ideal answer is low."
            )
        if concept_score < 70:
            reasons.append("Several important concepts were missing.")
        if attention_score < 70:
            reasons.append("The student's attention level was inconsistent.")
        if confidence_score < 70:
            reasons.append("Multiple long pauses reduced speech confidence.")
        if mis_count > 0:
            reasons.append(f"{mis_count} misconception(s) were detected.")
        if knowledge_gap > 0:
            reasons.append(f"{knowledge_gap} important concepts were not explained.")

        if not reasons:
            reasons.append("The student demonstrated strong understanding across the evaluated signals.")

        return {
            "understanding_score": round(score, 2),
            "performance": level,
            "reasons": reasons,
        }
