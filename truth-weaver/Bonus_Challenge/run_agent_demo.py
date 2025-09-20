#!/usr/bin/env python3
# bonus_agentic_flow/run_agent_demo.py
"""
Regex-only agentic flow demo with deepened reasoning.
Detects rich interview signals from transcripts and maps them to nuanced interviewer actions.

Improvements:
- Larger set of linguistic + behavioral cues
- More detailed signal categories (ownership, evasion, exaggeration, contradictions)
- Richer recommended interviewer actions
- Designed for broader coverage and reasoning without AI fallback
"""

import os
import re
import json

# --- Signal detection (regex only) ---
def detect_signals_regex(transcript: str, session_id: str):
    """
    Detect multiple interview signals using regex patterns and heuristic rules.
    Returns list of signals like ["hesitation", "contradiction", ...].
    """
    signals = []
    t = transcript.lower()

    # === Communication / Clarity ===
    if re.search(r"\b(maybe|probably|i think|not sure|kind of|sort of|guess)\b", t):
        signals.append("hesitation")

    if len(re.findall(r"\b(uh|um|hmm|like|you know)\b", t)) >= 3:
        signals.append("filler_words")

    if re.search(r"\b(well actually|no wait|i mean|what i meant)\b", t):
        signals.append("backtracking")

    if re.search(r"\b(off the top of my head|hard to say|cannot recall|don’t remember)\b", t):
        signals.append("memory_gap")

    # === Confidence spectrum ===
    if re.search(r"\b(seasoned|expert|mastered|handled full life cycle|always say|never wrong|world-class)\b", t):
        signals.append("overconfidence")

    if re.search(r"\b(not confident|unsure|newbie|still learning|don’t know)\b", t):
        signals.append("underconfidence")

    # === Ownership vs Disclaimers ===
    if re.search(r"\b(i just|i mostly|i only|i copied|i reused|i watched|not really|helped|supported)\b", t):
        signals.append("admission")

    if re.search(r"\b(contributed|assisted|participated|part of|helped with)\b", t):
        signals.append("weak_ownership")

    # === Contradictions / inconsistencies ===
    if re.search(r"\b(no actually|but then|however|in reality|later realized|on second thought)\b", t):
        signals.append("contradiction")

    if re.search(r"\b(\d+\s+years).*(internship|student|just started)\b", t):
        signals.append("experience_inconsistency")

    if re.search(r"\b(this year|few months ago|last month)\b.*\b(always|for years)\b", t):
        signals.append("timeline_contradiction")

    # === Leadership / Role inflation ===
    if re.search(r"\b(i led|i managed|i headed|i was responsible|i coordinated|i supervised)\b", t):
        signals.append("leadership_claim")

    if re.search(r"\b(coordinated is different from led|i didn’t really lead|was more like helping)\b", t):
        signals.append("leadership_inconsistency")

    # === Vagueness / Avoidance ===
    if re.search(r"\b(many years|long time|lots of experience|quite a while|some time|various projects|different things)\b", t):
        signals.append("vagueness")

    if re.search(r"\b(it depends|context-specific|case by case)\b", t):
        signals.append("evasion")

    # === Emotional / Behavioral cues ===
    if re.search(r"\b(sigh|laugh|cry|chuckle|pause|hmm|hahaha|nervous|anxious|upset)\b", t):
        signals.append("emotion")

    if re.search(r"\b(to be honest|frankly|truth is|honestly)\b", t):
        signals.append("defensive_marker")

    # === Buzzword inflation ===
    if len(re.findall(r"\b(ai|blockchain|synergy|disruption|cutting edge|cloud native)\b", t)) >= 2:
        signals.append("buzzword_inflation")

    # === Over-detailing (signal of overcompensation) ===
    if len(t.split()) > 250:  # very long response in one session
        signals.append("over_detailing")

    return list(set(signals))  # deduplicate


# --- Map signals to recommended interviewer actions ---
SIGNAL_ACTIONS = {
    "hesitation": "Ask for a concrete example to anchor their response.",
    "filler_words": "Encourage a pause: 'Take your time, no rush.'",
    "backtracking": "Request a clear restatement to check consistency.",
    "memory_gap": "Ask if there’s a specific project that could help recall.",
    "overconfidence": "Request evidence, metrics, or peer validation.",
    "underconfidence": "Probe supportive experiences to check hidden strengths.",
    "admission": "Drill into what was personally done vs. observed.",
    "weak_ownership": "Clarify specific role and responsibilities in the project.",
    "contradiction": "Highlight the inconsistency and ask for reconciliation.",
    "experience_inconsistency": "Clarify timeline of internships vs. years claimed.",
    "timeline_contradiction": "Pinpoint exact timeline to resolve conflicting claims.",
    "leadership_claim": "Ask about scope, authority, and measurable outcomes.",
    "leadership_inconsistency": "Challenge: 'You said led, but also just coordinated — which is accurate?'",
    "vagueness": "Push for numbers, names, and measurable results.",
    "evasion": "Reframe: 'If you had to choose one concrete example, what would it be?'",
    "emotion": "Acknowledge and ease pressure: 'Take your time, I’d like details.'",
    "defensive_marker": "Reassure and then re-ask the question neutrally.",
    "buzzword_inflation": "Ask how the buzzword was concretely applied in practice.",
    "over_detailing": "Steer back: 'Summarize in 2–3 key points.'"
}


# --- Main execution ---
if __name__ == "__main__":
    transcripts_path = "../transcribed.txt"

    if not os.path.exists(transcripts_path):
        print(f"[!] Transcript file not found: {transcripts_path}")
        exit(1)

    with open(transcripts_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    # Split transcript into sessions (now marked by === filename ===)
    raw_sessions = re.split(r"={3,}\s*([^\n=]+)\s*={3,}", full_text)
    sessions = []
    for i in range(1, len(raw_sessions), 2):
        filename = raw_sessions[i].strip()
        transcript = raw_sessions[i+1].strip()
        if transcript:
            sessions.append((filename, transcript))

    timeline = []
    for i, (fname, session_text) in enumerate(sessions, start=1):
        session_id = fname  # use audio filename as session id
        excerpt = session_text[:300] + ("..." if len(session_text) > 300 else "")

        signals = detect_signals_regex(session_text, session_id)

        actions = []
        for sig in signals:
            recommended = SIGNAL_ACTIONS.get(sig, "Note and continue probing later.")
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
        "summary": "Interview completed. Candidate exhibited a range of signals (clarity, confidence, ownership, consistency) requiring targeted follow-ups."
    })

    # Save JSON log
    with open("agentic_log.json", "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    print("[+] Agentic log saved to agentic_log.json")
