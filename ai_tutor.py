import json

import ollama


class AITutor:
    def generate_reference(self, topic):
        prompt = f"""
You are an expert computer science teacher.

Generate a JSON object for the topic "{topic}" with the following fields:

- topic_explanation: a simple explanation for first-year engineering students.
- real_world_example: a real-world example.
- applications: a list of three short application statements.
- ideal_answer: a strong answer the student should aim for.
- concepts: a list of the main concepts the student should mention.
- keywords: a list of important keywords.
- misconceptions: a list of common misconceptions about the topic.

Return only valid JSON with these keys.
"""

        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "num_gpu": 0,
                "low_vram": True,
            },
        )

        content = response["message"]["content"]
        try:
            reference = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Unable to parse reference JSON from Ollama response. "
                f"Response content was: {content!r}"
            ) from exc

        return reference

    def generate_question(self, topic):
        prompt = f"""
Generate one interview-style conceptual question about {topic}.

The question should test understanding instead of memorization.

Return only the question.
"""

        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "num_gpu": 0,
                "low_vram": True,
            },
        )

        return response["message"]["content"]
