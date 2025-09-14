import os
import re
import json
from agent_flow import AgenticFlow

# --- Helper: detect signals from transcript ---
def detect_signals(transcript, session_id):
    signals = []

    # Overconfidence markers
    if re.search(r"\b(seasoned|mastered|personally responsible|wrote all)\b", transcript, re.I):
        signals.append("overconfidence")

    # Hesitation markers
    if re.search(r"\bmaybe\b|\bprobably\b|\bjust\b", transcript, re.I):
        signals.append("hesitation")

    # Contradictions across sessions
    if session_id == "session1" and "seasoned" in transcript.lower():
        signals.append("contradiction")
    if session_id == "session5" and "internship" in transcript.lower():
        signals.append("contradiction")

    # Admissions of weakness
    if re.search(r"\bi just\b|\bnot\b|\bwatched\b", transcript.lower()):
        signals.append("emotion_sobbing")

    return signals


if __name__ == "__main__":
    transcripts_folder = "../transcripts"
    agent = AgenticFlow()

    timeline = []

    for fname in sorted(os.listdir(transcripts_folder)):
        if fname.endswith(".txt"):
            session_id = fname.replace(".txt", "")
            with open(os.path.join(transcripts_folder, fname), "r", encoding="utf-8") as f:
                transcript = f.read().strip()

            signals = detect_signals(transcript, session_id)
            actions = []

            for sig in signals:
                action = agent.handle_signal(sig)
                actions.append({"signal": sig, "state": agent.state, "action": action})

            timeline.append({
                "session": session_id,
                "transcript": transcript,
                "signals": signals,
                "actions": actions,
                "final_state": agent.state
            })

    # Add wrap-up
    final_action = agent.handle_signal("time_up")
    timeline.append({
        "session": "wrap_up",
        "signals": ["time_up"],
        "actions": [{"signal": "time_up", "state": agent.state, "action": final_action}],
        "final_state": agent.state,
        "summary": final_action
    })

    # Save JSON log
    with open("agentic_log.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    print("Agentic log saved as agentic_log.json")
