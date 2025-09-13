import subprocess
import json
import os
import re
import math

TRANSCRIPT_DIR = "transcripts"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- OLLAMA HELPER -------------------
def run_ollama(prompt, model="llama3"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

def extract_json(text):
    """Return the first balanced {...} JSON substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def repair_json(json_str):
    """Fix common JSON issues like trailing commas or missing commas."""
    json_str = re.sub(r"\},\s*\{", "}, {", json_str)  # fix missing commas
    json_str = json_str.replace(",]", "]")           # remove trailing commas
    return json_str

# ------------------- EXPERIENCE HELPERS -------------------
def parse_years_from_claim(claim):
    """
    Extract a numeric years value from a claim string.
    Returns float years or None.
    """
    claim = claim.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*years?", claim)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*yrs?", claim)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*months?", claim)
    if m:
        return round(float(m.group(1)) / 12.0, 2)
    m = re.search(r"(\d+)\s*years?\s*(?:and|,)?\s*(\d+)\s*months?", claim)
    if m:
        years = float(m.group(1)) + float(m.group(2)) / 12.0
        return round(years, 2)
    if "a few" in claim:
        return 3.0
    if "several" in claim:
        return 4.0
    if "many years" in claim or "long time" in claim:
        return None
    return None

def format_experience_years(years_min, years_max, prefer_range_threshold=1.0):
    """
    Format final experience string according to rules:
    - if min and max within threshold -> return nearest integer "N years"
    - otherwise return "MIN-MAX years"
    """
    if years_min is None and years_max is None:
        return None
    if years_min is None:
        years_min = years_max
    if years_max is None:
        years_max = years_min
    if abs(years_max - years_min) <= prefer_range_threshold:
        val = round((years_min + years_max) / 2.0)
        return f"{int(val)} years"
    low = math.floor(years_min)
    high = math.ceil(years_max)
    if low == high:
        return f"{low} years"
    return f"{low}-{high} years"

# ------------------- PROMPT BUILDER -------------------
def build_prompt(transcripts, shadow_id="agent_x"):
    text = "\n".join([f"Session {i+1}: {t}" for i, t in enumerate(transcripts)])
    return f"""
You are Truth Weaver AI. Your job is to ANALYZE the provided interview/transcript SESSIONS and produce ONLY ONE JSON object (no explanations, no markdown, no extra text). The JSON must be syntactically valid and follow the schema exactly. If a field cannot be determined confidently, set it to null and explain in 'notes'.

--- SCHEMA ---
{{
  "shadow_id": "{shadow_id}",
  "revealed_truth": {{
    "programming_experience": "...",
    "programming_language": "...",
    "skill_mastery": "...",
    "leadership_claims": "...",
    "team_experience": "...",
    "skills and other keywords": ["..."]
  }},
  "deception_patterns": [
    {{
      "lie_type": "...",
      "contradictory_claims": ["...", "..."],
      "sessions": [1, 3],
      "confidence": 0.0
    }}
  ],
  "field_confidence": {{
    "programming_experience": 0.0,
    "programming_language": 0.0,
    "skill_mastery": 0.0,
    "leadership_claims": 0.0,
    "team_experience": 0.0,
    "skills and other keywords": 0.0
  }},
  "evidence": {{
    "programming_experience": [
      {{ "session": 1, "claim": "I have 6 years of experience" }},
      {{ "session": 3, "claim": "I've been coding for 3 years" }}
    ]
  }},
  "notes": "..."
}}

--- RULES ---
1. Output ONLY the JSON object.
2. PROGRAMMING_EXPERIENCE: parse numeric years/months; if close, return "N years"; if far apart, return "MIN-MAX years".
3. PROGRAMMING_LANGUAGE: lowercase primary language; null if none clear.
4. SKILL_MASTERY: one of ["beginner","intermediate","advanced"].
5. LEADERSHIP_CLAIMS: "fabricated" if contradictions, "genuine" if consistent, "uncertain" if unclear.
6. TEAM_EXPERIENCE: "individual contributor", "team player", or "uncertain".
7. SKILLS: technical keywords only, title-cased, deduplicated.
8. DECEPTION_PATTERNS: list contradictions with sessions + confidence score.
9. CONFIDENCE: 0-1 per field.
10. EVIDENCE: list raw claims with session numbers.
11. Use null for unknowns, never hallucinate facts.
12. All keys exactly as shown.

--- SESSIONS ---
{text}

Return strictly one valid JSON object.
"""

# ------------------- MAIN -------------------
if __name__ == "__main__":
    transcripts = []
    for file in sorted(os.listdir(TRANSCRIPT_DIR)):
        with open(os.path.join(TRANSCRIPT_DIR, file), "r", encoding="utf-8") as f:
            transcripts.append(f.read())

    prompt = build_prompt(transcripts, shadow_id="phoenix_2024")
    response = run_ollama(prompt)

    json_str = extract_json(response)
    if not json_str:
        print("[!] Could not find JSON in model response. Raw output:")
        print(response)
        exit()

    # Try parsing directly, else repair
    try:
        parsed = json.loads(json_str)
    except Exception:
        json_str = repair_json(json_str)
        try:
            parsed = json.loads(json_str)
        except Exception as e:
            print("[!] JSON parsing failed even after repair:", e)
            print("Raw extracted string:\n", json_str)
            exit()

    out_file = os.path.join(OUTPUT_DIR, "phoenix_2024.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    print(f"[+] JSON saved: {out_file}")

    # -------- POST-PROCESSING NORMALIZATION --------
    truth = parsed.get("revealed_truth", {})

    # Normalize programming experience
    exp_claims = []
    for ev in parsed.get("evidence", {}).get("programming_experience", []):
        years = parse_years_from_claim(ev.get("claim", ""))
        if years is not None:
            exp_claims.append(years)
    if exp_claims:
        truth["programming_experience"] = format_experience_years(min(exp_claims), max(exp_claims))

    # Ensure experience is string
    if isinstance(truth.get("programming_experience"), (int, float)):
        truth["programming_experience"] = f"{truth['programming_experience']} years"

    # Normalize programming language
    if isinstance(truth.get("programming_language"), str):
        truth["programming_language"] = truth["programming_language"].lower()

    # Fix leadership claims if contradictions exist
    if parsed.get("deception_patterns"):
        if truth.get("leadership_claims") == "genuine":
            for dp in parsed["deception_patterns"]:
                if "leadership" in " ".join(dp.get("contradictory_claims", [])).lower():
                    truth["leadership_claims"] = "fabricated"

    # Save normalized JSON again
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"[+] Normalized JSON updated: {out_file}")
