import os
import whisper
from pydub import AudioSegment
from transformers import pipeline
import numpy as np

# Input / Output directories
INPUT_DIR = "data"   # preprocessed audios
OUTPUT_DIR = "transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Whisper
model = whisper.load_model("small")

# Load Hugging Face Emotion Recognition
print("[+] Loading emotion recognition model...")
emotion_pipeline = pipeline("audio-classification", model="superb/hubert-large-superb-er")

# ------------------- Emotion Mapper -------------------
def map_emotion(label, energy):
    label = label.lower()
    if label == "happy":
        return "[sarcastic laugh]" if energy < -25 else "[laughing]"
    elif label == "sad":
        return "[sigh]" if energy > -30 else "[sobbing]"
    elif label == "angry":
        return "[shouting]"
    elif label == "fearful":
        return "[whispering]"
    else:
        return ""  # neutral → no special cue

# ------------------- Feature Extraction -------------------
def detect_emotion(segment_file, segment_audio):
    try:
        result = emotion_pipeline(segment_file)
        top_label = result[0]["label"].lower()
        # Use loudness to refine emotion
        energy = segment_audio.dBFS
        cue = map_emotion(top_label, energy)
        return cue
    except Exception:
        return ""

# ------------------- Enrichment -------------------
def enrich_transcript(result, audio_file):
    transcript = []
    audio = AudioSegment.from_file(audio_file)

    for seg in result["segments"]:
        text = seg["text"].strip()

        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]

        # Export for emotion detection
        segment_file = "temp_segment.wav"
        segment_audio.export(segment_file, format="wav")

        # Detect emotion
        cue = detect_emotion(segment_file, segment_audio)
        if cue:
            text = f"{cue} {text}"

        # Insert pauses for long gaps
        if seg.get("no_speech_prob", 0) > 0.6:
            text += " ..."

        # Replace inaudible markers
        text = text.replace("<|inaudible|>", "[inaudible]")

        transcript.append(text)

    return " ".join(transcript)

# ------------------- Main -------------------
def transcribe_audio(file_path, session_name):
    print(f"[+] Transcribing {file_path} ...")
    result = model.transcribe(file_path, word_timestamps=True)

    transcript = enrich_transcript(result, file_path)

    # Return transcript instead of writing individual files
    return f"\n\n=== {session_name.upper()} ===\n{transcript}"


if __name__ == "__main__":
    all_transcripts = []

    for i, file in enumerate(sorted(os.listdir(INPUT_DIR)), start=1):
        if file.endswith(".wav") or file.endswith(".mp3"):
            session_name = f"Session {i}"
            session_transcript = transcribe_audio(os.path.join(INPUT_DIR, file), session_name)
            all_transcripts.append(session_transcript)

    # Save combined transcript
    out_file = os.path.join(OUTPUT_DIR, "transcript.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_transcripts))

    print(f"[+] Final transcript saved: {out_file}")

