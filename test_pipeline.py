import json
from verify import run_unified_verification


def main():
    reference = {
        "topic_explanation": "Binary search finds an item in a sorted array by repeatedly halving the search range.",
        "ideal_answer": "Binary search works on sorted arrays by repeatedly halving the search range and comparing the target to the middle element.",
        "concepts": {"Binary Search": 10, "Sorted Array": 8, "Middle Element": 6},
        "misconceptions": ["Binary search works on unsorted arrays."],
    }

    transcript = (
        "Binary search repeatedly divides the search area into halves, checking the middle element each time to find the target."
    )

    video_path = ""  # no video for headless test
    timestamps = []

    res = run_unified_verification(reference, video_path, transcript, timestamps)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
