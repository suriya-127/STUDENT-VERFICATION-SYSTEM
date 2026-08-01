import json
import os
import shutil

import whisper


class WhisperModel:
    def __init__(self):
        self._ensure_ffmpeg()
        self.model = whisper.load_model("base")

    def _ensure_ffmpeg(self):
        if shutil.which("ffmpeg"):
            return

        candidate_dirs = [
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
            r"C:\Users\jayas\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin",
            os.path.join(
                os.path.expandvars(r"%LOCALAPPDATA%"),
                "Microsoft",
                "WinGet",
                "Packages",
                "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
                "ffmpeg-8.1.2-full_build",
                "bin",
            ),
            os.path.expandvars(r"%ProgramFiles%\ffmpeg\bin"),
            os.path.expandvars(r"%ProgramFiles(x86)%\ffmpeg\bin"),
            os.path.expanduser(r"~\AppData\Local\Programs\ffmpeg\bin"),
            os.path.expanduser(r"~\OneDrive\Documents\PROJECTS\VERFICATION SYSTEM\bin"),
            r"C:\Users\jayas\OneDrive\Documents\PROJECTS\VERFICATION SYSTEM\bin",
        ]

        for candidate_dir in candidate_dirs:
            if os.path.isdir(candidate_dir):
                candidate_exe = os.path.join(candidate_dir, "ffmpeg.exe")
                if os.path.exists(candidate_exe):
                    os.environ["PATH"] = candidate_dir + os.pathsep + os.environ.get("PATH", "")
                    return

                for dirpath, _, filenames in os.walk(candidate_dir):
                    if "ffmpeg.exe" in filenames:
                        os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")
                        return

        raise FileNotFoundError(
            "ffmpeg executable not found. Install FFmpeg and add it to PATH, "
            "or update whisper_model.py with the correct ffmpeg.exe path."
        )

    def transcribe(self, video_path):
        result = self.model.transcribe(video_path, word_timestamps=True)
        transcript = result["text"]
        timestamps = []

        for segment in result.get("segments", []):
            if "words" in segment:
                for word in segment["words"]:
                    timestamps.append(
                        {
                            "word": word["word"].strip(),
                            "start": word["start"],
                            "end": word["end"],
                        }
                    )

        return {
            "transcript": transcript,
            "timestamps": timestamps,
            "raw_result": result,
        }

    def save_outputs(self, transcript, timestamps):
        with open("outputs/transcript.txt", "w", encoding="utf8") as f:
            f.write(transcript)

        with open("outputs/timestamps.json", "w", encoding="utf8") as f:
            json.dump(timestamps, f, indent=4)
