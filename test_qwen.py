import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Explain Binary Search in simple English with a real-world example."
        }
    ]
)

print(response["message"]["content"])
