import os
import re
import tempfile
import subprocess

from run_model import chat, conversation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "microsoft/VibeVoice-Realtime-0.5B")
VIBEVOICE_REPO_PATH = os.environ.get(
    "VIBEVOICE_REPO_PATH", os.path.join(SCRIPT_DIR, "VibeVoice")
)
VIBEVOICE_PYTHON = os.environ.get("VIBEVOICE_PYTHON", "python")
TTS_SPEAKER_NAME = os.environ.get("TTS_SPEAKER_NAME", "Carter")


def remove_emojis(text: str) -> str:
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


def extract_generation_summary(output: str) -> str:
    start_marker = "==================================================\nGENERATION SUMMARY"
    start = output.find(start_marker)

    if start == -1:
        return ""

    return output[start:].strip()


def synthesize_with_vibevoice_realtime(text: str, output_dir: str) -> str:
    if not os.path.isdir(VIBEVOICE_REPO_PATH):
        raise FileNotFoundError(
            "VibeVoice repo path not found. Set VIBEVOICE_REPO_PATH to your local "
            f"VibeVoice directory. Current value: {VIBEVOICE_REPO_PATH}"
        )

    txt_path = os.path.join(output_dir, "reply.txt")
    reply_wav = os.path.join(VIBEVOICE_REPO_PATH, "outputs", "reply_generated.wav")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    if os.path.exists(reply_wav):
        os.remove(reply_wav)

    cmd = [
        VIBEVOICE_PYTHON,
        "demo/realtime_model_inference_from_file.py",
        "--model_path", TTS_MODEL_ID,
        "--txt_path", txt_path,
        "--speaker_name", TTS_SPEAKER_NAME,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=VIBEVOICE_REPO_PATH,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("\nTTS failed.")
        print("\n--- TTS STDOUT ---")
        print(e.stdout if e.stdout else "(no stdout)")
        print("--- END TTS STDOUT ---")
        print("\n--- TTS STDERR ---")
        print(e.stderr if e.stderr else "(no stderr)")
        print("--- END TTS STDERR ---\n")
        raise

    summary = extract_generation_summary(result.stdout)
    if summary:
        print(summary)

    if not os.path.exists(reply_wav):
        raise FileNotFoundError(
            f"TTS finished, but expected WAV was not found: {reply_wav}"
        )

    return reply_wav


if __name__ == "__main__":
    print("Typed chat + TTS ready. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        response, updated_conversation, feeling = chat(user_input, conversation)
        conversation[:] = updated_conversation

        clean_response = remove_emojis(response)

        print(f"[detected user emotion: {feeling}]")
        print(f"Assistant: {clean_response}")

        with tempfile.TemporaryDirectory() as tmpdir:
            reply_wav = synthesize_with_vibevoice_realtime(clean_response, tmpdir)
            print(f"TTS audio saved at: {reply_wav}")