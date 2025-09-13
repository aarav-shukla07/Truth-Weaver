import os
import re
from agent_flow import AgenticFlow

def detect_signals(transcript, session_id):
    signals = []

    # --- Overconfidence / inflated claims ---
    if re.search(r"\b(seasoned|mastered|personally responsible|wrote all)\b", transcript, re.I):
        signals.append("overconfidence")

    # --- Hesitation / vague ---
    if re.search(r"\bmaybe\b|\bprobably\b|\bjust\b", transcript, re.I):
        signals.append("hesitation")

    # --- Contradictions between sessions ---
    if session_id == "session1" and "seasoned" in transcript.lower():
        signals.append("contradiction")  # later they admit internship
    if session_id == "session5" and "internship" in transcript.lower():
        signals.append("contradiction")

    # --- Admission of weakness ---
    if re.search(r"\bi just\b|\bnot\b|\bwatched\b", transcript.lower()):
        signals.append("emotion_sobbing")  # treat as "confession moment"

    return signals


if __name__ == "__main__":
    transcripts_folder = "../transcripts"
    agent = AgenticFlow()

    for fname in sorted(os.listdir(transcripts_folder)):
        if fname.endswith(".txt"):
            session_id = fname.replace(".txt", "")
            print(f"\n Processing {fname}")
            with open(os.path.join(transcripts_folder, fname), "r", encoding="utf-8") as f:
                transcript = f.read().strip()

            print("Transcript:", transcript)
            signals = detect_signals(transcript, session_id)
            print("Detected signals:", signals)

            for sig in signals:
                action = agent.handle_signal(sig)
                print(f" Signal: {sig} | State: {agent.state} | Action: {action}")

    # Wrap up
    final_action = agent.handle_signal("time_up")
    print(f"\n Interview ended | Action: {final_action}")
