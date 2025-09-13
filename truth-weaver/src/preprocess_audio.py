import os
from pydub import AudioSegment, effects
import librosa
import noisereduce as nr
import soundfile as sf

INPUT_DIR = "data"
CLEAN_DIR = "cleaned_data"
os.makedirs(CLEAN_DIR, exist_ok=True)

def preprocess_audio(in_file, out_file):
    print(f"[+] Preprocessing {in_file}")

    # Load audio
    audio = AudioSegment.from_file(in_file)

    # Convert to mono and normalize loudness
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)  # 16kHz is best for Whisper
    audio = effects.normalize(audio)

    # Save temporary WAV for noise reduction
    temp_file = out_file.replace(".wav", "_temp.wav")
    audio.export(temp_file, format="wav")

    # Load for noise reduction with librosa
    y, sr = librosa.load(temp_file, sr=16000)
    reduced = nr.reduce_noise(y=y, sr=sr)

    # Save cleaned file
    sf.write(out_file, reduced, sr)

    # Cleanup temp file
    os.remove(temp_file)

    print(f"[+] Saved cleaned audio → {out_file}")

if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".mp3") or file.endswith(".wav"):
            base = os.path.splitext(file)[0]
            out_file = os.path.join(CLEAN_DIR, f"{base}.wav")
            preprocess_audio(os.path.join(INPUT_DIR, file), out_file)
