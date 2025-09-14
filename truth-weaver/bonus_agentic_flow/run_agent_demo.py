#!/usr/bin/env python3
# bonus_agentic_flow/run_agent_demo.py
"""
Regex-only agentic flow demo.
Detects rich interview signals from transcripts and maps them to interviewer actions.
"""

import os
import re
import json

# --- Signal detection (regex only) ---
def detect_signals_regex(transcript: str, session_id: str):
    """
    Detect multiple interview signals using regex patterns.
    Returns list of signals like ["hesitation", "contradiction", ...].
    """
    signals = []
    t = transcript.lower()

    # Hesitation markers
    if re.search(r"\b(maybe|probably|i think|not sure|kind of|sort of|guess)\b", t):
        signals.append("hesitation")

    # Overconfidence / exaggeration
    if re.search(r"\b(seasoned|expert|mastered|handled full life cycle|always say|never wrong)\b", t):
        signals.append("overconfidence")

    # Admissions / disclaimers
    if re.search(r"\b(i just|i mostly|i only|i copied|i reused|i watched|not really)\b", t):
        signals.append("admission")

    # Contradictions
    if re.search(r"\b(no actually|but then|however|in reality|later realized)\b", t):
        signals.append("contradiction")

    # Leadership claims
    if re.search(r"\b(i led|i managed|i headed|i was responsible|i coordinated)\b", t):
        signals.append("leadership_claim")

    # Leadership self-undercut
    if re.search(r"\b(coordinated is different from led|i didn’t really lead)\b", t):
        signals.append("leadership_inconsistency")

    # Vagueness
    if re.search(r"\b(many years|long time|lots of experience|quite a while|some time)\b", t):
        signals.append("vagueness")

    # Backtracking
    if re.search(r"\b(well actually|no wait|i mean|what i meant)\b", t):
        signals.append("backtracking")

    # Emotional cues
    if re.search(r"\b(sigh|laugh|cry|chuckle|pause|hmm|hahaha)\b", t):
        signals.append("emotion")

    # Filler words (communication clarity issues)
    if len(re.findall(r"\b(uh|um|hmm|like)\b", t)) >= 3:
        signals.append("filler_words")

    # Inconsistencies with experience
    if "internship" in t and re.search(r"\b\d+\s+years\b", t):
        signals.append("experience_inconsistency")

    return list(set(signals))  # deduplicate


# --- Map signals to recommended interviewer actions ---
SIGNAL_ACTIONS = {
    "hesitation": "Ask for a concrete example.",
    "overconfidence": "Request evidence or metrics to support the claim.",
    "admission": "Probe deeper into what exactly was done personally.",
    "contradiction": "Clarify the inconsistency.",
    "leadership_claim": "Ask about scope and team size to validate leadership.",
    "leadership_inconsistency": "Challenge the difference between 'led' vs 'coordinated'.",
    "vagueness": "Request specific numbers or timelines.",
    "backtracking": "Ask the candidate to restate clearly.",
    "emotion": "Acknowledge and refocus: 'Take your time, can you clarify?'",
    "filler_words": "Encourage concise and structured response.",
    "experience_inconsistency": "Ask for clarification on internship vs years of experience."
}


# --- Main execution ---
if __name__ == "__main__":
    transcripts_path = "../transcripts/transcript.txt"

    if not os.path.exists(transcripts_path):
        print(f"[!] Transcript file not found: {transcripts_path}")
        exit(1)

    with open(transcripts_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    # Split transcript into sessions (heuristic: blank lines or "Session N")
    raw_sessions = re.split(r"(?:\n\s*\n|Session\s*\d+:)", full_text)
    sessions = [s.strip() for s in raw_sessions if s.strip()]

    timeline = []
    for i, session_text in enumerate(sessions, start=1):
        session_id = f"Session{i}"
        excerpt = session_text[:300] + ("..." if len(session_text) > 300 else "")

        signals = detect_signals_regex(session_text, session_id)

        actions = []
        for sig in signals:
            recommended = SIGNAL_ACTIONS.get(sig, "Note and continue.")
            actions.append({
                "signal": sig,
                "recommended_action": recommended
            })

        recommended_summary = actions[0]["recommended_action"] if actions else "No action required."

        timeline.append({
            "order": i,
            "session": session_id,
            "transcript_excerpt": excerpt,
            "signals": signals,
            "actions": actions,
            "recommended_action_summary": recommended_summary
        })

    # Add wrap-up
    timeline.append({
        "session": "wrap_up",
        "signals": ["time_up"],
        "actions": [],
        "final_state": "wrap_up",
        "summary": "Interview completed. Candidate showed mixed signals requiring follow-up on specifics."
    })

    # Save JSON log
    with open("agentic_log.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    print("[+] Agentic log saved to agentic_log.json")
