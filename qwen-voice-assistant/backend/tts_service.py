"""
TTS Service using Microsoft VibeVoice for speech synthesis.
"""

import os
import re
import subprocess
import tempfile
import glob
from typing import Optional
import uuid
import sys


class TTSService:
    def __init__(self):
        self.repo_path = os.environ.get(
            "VIBEVOICE_REPO_PATH",
            os.path.join(os.path.dirname(__file__), "..", "VibeVoice"),
        )
        self.python_path = self._resolve_python_path()
        self.model_id = os.environ.get(
            "TTS_MODEL_ID", "microsoft/VibeVoice-Realtime-0.5B"
        )
        self.speaker_name = os.environ.get("TTS_SPEAKER_NAME", "Carter")
        self.timeout_seconds = int(os.environ.get("TTS_TIMEOUT_SECONDS", "240"))
        self.speaker_aliases = {
            "sophia": "Emma",
            "michael": "Mike",
        }

        self.output_dir = os.path.join(
            os.path.dirname(__file__), "..", "audio_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self._available = os.path.isdir(self.repo_path)
        if not self._available:
            print(f"VibeVoice not found at {self.repo_path}. TTS disabled.")

    def _resolve_python_path(self) -> str:
        env_python = os.environ.get("VIBEVOICE_PYTHON", "").strip()
        if env_python:
            return env_python

        project_venv_python = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset_CUDA",
                ".venv",
                "Scripts",
                "python.exe",
            )
        )
        if os.path.exists(project_venv_python):
            return project_venv_python

        if sys.executable:
            return sys.executable

        return "python"

    def is_available(self) -> bool:
        return self._available

    def remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text).strip()

    def synthesize(self, text: str, speaker: Optional[str] = None) -> str:
        if not self._available:
            raise RuntimeError("VibeVoice is not available")

        clean_text = self.remove_emojis(text)
        speaker = speaker or self.speaker_name
        speaker = self.speaker_aliases.get(speaker.strip().lower(), speaker)

        output_filename = f"{uuid.uuid4().hex}.wav"
        output_path = os.path.join(self.output_dir, output_filename)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(clean_text)
            txt_path = f.name

        try:
            txt_stem = os.path.splitext(os.path.basename(txt_path))[0]
            with tempfile.TemporaryDirectory(prefix="vibevoice_out_") as run_output_dir:
                cmd = [
                    self.python_path,
                    "demo/realtime_model_inference_from_file.py",
                    "--model_path", self.model_id,
                    "--txt_path", txt_path,
                    "--speaker_name", speaker,
                    "--output_dir", run_output_dir,
                ]

                try:
                    subprocess.run(
                        cmd,
                        cwd=self.repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as e:
                    raise RuntimeError(
                        f"VibeVoice inference timed out after {self.timeout_seconds}s. "
                        "Increase TTS_TIMEOUT_SECONDS if model warm-up is slow."
                    ) from e
                except subprocess.CalledProcessError as e:
                    stdout = (e.stdout or "").strip()
                    stderr = (e.stderr or "").strip()
                    details = "\n".join(part for part in [stdout, stderr] if part)
                    if not details:
                        details = f"exit code {e.returncode}"
                    raise RuntimeError(f"VibeVoice inference failed: {details}") from e

                candidates = sorted(glob.glob(os.path.join(run_output_dir, "*_generated.wav")))
                expected_wav = os.path.join(run_output_dir, f"{txt_stem}_generated.wav")
                selected_wav = expected_wav if os.path.exists(expected_wav) else (candidates[0] if candidates else None)
                if not selected_wav:
                    raise FileNotFoundError(
                        f"TTS completed but no generated wav found in: {run_output_dir}"
                    )

                import shutil
                shutil.copy(selected_wav, output_path)

        finally:
            if os.path.exists(txt_path):
                os.remove(txt_path)

        return output_path

    def get_available_speakers(self) -> list:
        return ["Carter", "Emma", "Michael", "Sophia"]

    def cleanup_old_files(self, max_age_hours: int = 1):
        import time
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath):
                file_age = now - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
