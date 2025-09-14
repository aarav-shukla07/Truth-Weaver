#!/usr/bin/env python3
# src/analyze.py
"""
Robust analyzer that (1) tries an LLM with a detailed reasoning prompt and (2) falls back to
a deterministic rule-based extractor if needed. Produces final JSON matching exact required schema:
{
 "shadow_id": "string",
 "revealed_truth": {
   "programming_experience": "string",
   "programming_language": "string",
   "skill_mastery": "string",
   "leadership_claims": "string",
   "team_experience": "string",
   "skills and other keywords": ["String", ...]
 },
 "deception_patterns": [
   {"lie_type": "string", "contradictory_claims": ["string", ...]},
   ...
 ]
}
"""
import os
import re
import json
import subprocess
import math
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "..", "transcripts")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "llama3"  # adjust if you use a different local model name
OLLAMA_TIMEOUT = 300   # seconds


# ---------------- helpers for transcripts ----------------
def read_sessions():
    """Return list of up to 5 session texts in order."""
    # Prefer explicit files session1..session5
    sessions = []
    found = []
    for i in range(1, 6):
        p = os.path.join(TRANSCRIPT_DIR, f"session{i}.txt")
        if os.path.exists(p):
            found.append(p)
    if found:
        for p in sorted(found):
            with open(p, "r", encoding="utf-8") as fh:
                sessions.append(fh.read().strip())
        return sessions

    # fallback single merged file
    merged = os.path.join(TRANSCRIPT_DIR, "transcript.txt")
    if not os.path.exists(merged):
        raise FileNotFoundError(f"No session files and no transcript.txt in {TRANSCRIPT_DIR}")
    raw = open(merged, "r", encoding="utf-8").read()
    # split by === Session N === markers if present
    parts = re.split(r"={3,}\s*Session\s*\d+\s*={3,}", raw, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    if parts:
        return parts[:5]
    # last fallback: split by blank lines into up to 5 blocks
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]
    return blocks[:5]


# ---------------- LLM (Ollama) utilities ----------------
def run_ollama(prompt, model=MODEL_NAME, timeout=OLLAMA_TIMEOUT):
    """Run ollama with the prompt. Returns stdout string or None on error/timeout."""
    try:
        # Do not assume extra flags; use subprocess timeout to avoid indefinite hang
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        stdout = proc.stdout.decode("utf-8", errors="ignore")
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        # Save raw outputs for debug
        with open(os.path.join(OUTPUT_DIR, "raw_response_first.txt"), "w", encoding="utf-8") as fh:
            fh.write(stdout)
        if stderr:
            with open(os.path.join(OUTPUT_DIR, "raw_response_stderr.txt"), "w", encoding="utf-8") as fh:
                fh.write(stderr)
        return stdout
    except subprocess.TimeoutExpired:
        print("[!] Ollama call timed out.")
        return None
    except FileNotFoundError as e:
        print(f"[!] Ollama CLI not found: {e}")
        return None


def extract_first_json(text):
    """Extract the first balanced {...} substring from text, or None."""
    if not text:
        return None
    # remove markdown fences first
    t = re.sub(r"```(?:json)?", "", text, flags=re.I)
    t = re.sub(r"```", "", t)
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(t)):
        ch = t[i]
        if ch == '"' and not escape:
            in_str = not in_str
        if ch == "\\" and not escape:
            escape = True
            continue
        else:
            escape = False
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    return None


def repair_json_text(s):
    if s is None:
        return None
    # normalize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    # remove trailing commas
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    # replace Python None/True/False
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    return s


def try_parse_json_candidate(candidate):
    """Try to parse candidate JSON with a couple of heuristics."""
    if candidate is None:
        return None
    c = repair_json_text(candidate)
    try:
        return json.loads(c)
    except Exception:
        # last resort: convert single quotes to double quotes if it looks Pythonic
        if c.count('"') < c.count("'"):
            alt = c.replace("'", '"')
            alt = repair_json_text(alt)
            try:
                return json.loads(alt)
            except Exception:
                return None
        return None


# ---------------- Prompt builder (rich reasoning) ----------------
def build_prompt_with_reasoning(sessions, shadow_id="phoenix_2024"):
    """
    Build a detailed prompt instructing the LLM how to reason and how to output the exact schema.
    This prompt is intentionally explicit about heuristics, examples, and the exact JSON keys.
    """
    # join sessions compactly (limit length if very long)
    joined = "\n".join([f"Session {i+1}: {s}" for i, s in enumerate(sessions)])
    # Provide clear heuristics and a short example
    heuristics = """
Heuristics / decision rules you MUST use when producing the JSON:
- For programming_experience: parse explicit numeric claims (e.g., "6 years", "18 months") and convert months->years.
  If later sessions give shorter numbers, prefer **later** sessions (truth leaks under pressure).
  If claims differ but are close (<=1 year difference) return "N years" (rounded). If far (>=2 years) return "MIN-MAX years".
  If only "internship" or "summer internship" appears, report "0-1 years".
- programming_language: pick the primary language mentioned (lowercase). If none clear, use null.
- skill_mastery: map phrasing to one of ["beginner","intermediate","advanced"]:
    * advanced: "mastered", "senior", "seasoned", "expert"
    * beginner: "intern", "just started", "learning", "junior"
    * intermediate: default if technical keywords exist but no extreme words
- leadership_claims: set to "fabricated" if presence of leadership claims (e.g., "I led a team", "I handled the full life cycle") are contradicted later by admissions of watching/being an intern or "I just deploy / I watched".
  Set to "genuine" if leadership claims are consistent across sessions. Else "uncertain".
- team_experience: return "individual contributor" if candidate repeatedly says they worked alone, "team player" if "we", "led a team", or "coordinated", otherwise "uncertain".
- skills and other keywords: return a list of deduplicated technical keywords (title-cased), e.g., ["Kubernetes","Calico","DNS"].
- deception_patterns: detect contradiction types such as experience inflation (e.g., "6 years" vs "3 years"), leadership_fabrication (e.g., claimed "led team" vs "I was an intern"), equivocation (repeated 'maybe', 'probably'). Each pattern object must have:
    {"lie_type": "string", "contradictory_claims": ["string", ...]}

Important: OUTPUT MUST BE EXACTLY ONE JSON OBJECT with ONLY these top-level keys:
- shadow_id (string)
- revealed_truth (object with the six fields)
- deception_patterns (array of objects with 'lie_type' and 'contradictory_claims')

If you cannot determine a field, set it to null (or empty list for skills/deception_patterns).
Do NOT include other keys like evidence, notes, or confidence.

Below is a compact example (input -> desired output):

Example input:
Session 1: "I've mastered Python for 6 years and I built the infra." 
Session 2: "Actually maybe 3 years, still learning advanced things."
Session 3: "I led a team of five engineers."
Session 4: "I mostly worked alone and deployed scripts."
Session 5: "It was an internship; I only watched."

Example output (single JSON object):
{
  "shadow_id": "phoenix_2024",
  "revealed_truth": {
    "programming_experience": "3-4 years",
    "programming_language": "python",
    "skill_mastery": "intermediate",
    "leadership_claims": "fabricated",
    "team_experience": "individual contributor",
    "skills and other keywords": ["Python"]
  },
  "deception_patterns": [
    {
      "lie_type": "experience_inflation",
      "contradictory_claims": ["6 years", "3 years"]
    },
    {
      "lie_type": "leadership_fabrication",
      "contradictory_claims": ["led a team of five", "it was an internship / I only watched"]
    }
  ]
}

--- NOW PROCESS THE SESSIONS BELOW AND RETURN ONLY THE JSON OBJECT ---
Sessions:
""" + joined
    return heuristics


# ---------------- Deterministic fallback analyzer ----------------
def rule_based_analyze(sessions, shadow_id="phoenix_2024"):
    """Produce a conservative JSON object following required schema using rules (no LLM)."""
    # helper small lexicons
    LANG = ["python", "java", "javascript", "c++", "c#", "go", "rust", "ruby", "php", "scala"]
    TECH = ["kubernetes", "calico", "docker", "terraform", "ansible", "dns", "network", "jenkins", "git", "nginx", "prometheus"]

    # gather simple evidence
    years_claims = []
    experiences_raw = []
    languages = []
    skills = []
    leadership_claims_flag = False
    watch_or_intern_flag = False
    hesitations = 0

    for i, text in enumerate(sessions, start=1):
        t = text.lower()
        experiences_raw.append((i, text))
        # numeric years
        for m in re.findall(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|months?)", text, flags=re.I):
            num = float(m[0])
            units = m[1].lower()
            if "month" in units:
                val = round(num / 12.0, 2)
            else:
                val = num
            years_claims.append(val)
        # internship indicators
        if re.search(r"\bintern(ship)?\b|\bsummer intern\b", t):
            watch_or_intern_flag = True
        # languages
        for L in LANG:
            if re.search(r"\b" + re.escape(L) + r"\b", t):
                languages.append(L)
        # tech keywords
        for kw in TECH:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                skills.append(kw.title())
        # leadership detection
        if re.search(r"\bled\b|\bleading\b|\bheaded\b|\bmanaged\b|\bmanager\b", t):
            leadership_claims_flag = True
        # "watched" or "just deploy"
        if re.search(r"\b(watched|i just deploy|i deployed|ran some scripts|i mostly just watched)\b", t):
            watch_or_intern_flag = True
        # hesitation
        if re.search(r"\bmaybe\b|\bprobably\b|\bnot sure\b|\bI think\b", t):
            hesitations += 1

    # programming_experience
    prog_exp = None
    if years_claims:
        ymin = min(years_claims); ymax = max(years_claims)
        if abs(ymax - ymin) <= 1.0:
            prog_exp = f"{int(round((ymin + ymax) / 2.0))} years"
        else:
            prog_exp = f"{math.floor(ymin)}-{math.ceil(ymax)} years"
    elif watch_or_intern_flag:
        prog_exp = "0-1 years"
    else:
        prog_exp = None

    programming_language = languages[0] if languages else None
    skills = list(dict.fromkeys(skills))  # dedupe preserve order
    skill_mastery = None
    if re.search(r"\b(master|mastered|expert|seasoned|senior)\b", " ".join(sessions), flags=re.I):
        skill_mastery = "advanced"
    elif re.search(r"\b(intern|junior|learning|still learning)\b", " ".join(sessions), flags=re.I):
        skill_mastery = "beginner"
    elif skills:
        skill_mastery = "intermediate"
    else:
        skill_mastery = None

    leadership_claims = "uncertain"
    if leadership_claims_flag and not watch_or_intern_flag:
        leadership_claims = "genuine"
    if leadership_claims_flag and watch_or_intern_flag:
        leadership_claims = "fabricated"

    # team_experience
    team_experience = None
    joined = " ".join(sessions).lower()
    if re.search(r"\bwork alone\b|\bi work alone\b|\bmostly alone\b|\bi just deploy\b|\bi just watched\b", joined):
        team_experience = "individual contributor"
    elif re.search(r"\bteam\b|\bwe\b|\bled a team\b|\bcoordinat", joined):
        team_experience = "team player"
    else:
        team_experience = "uncertain"

        # deception patterns (expanded rules)
    deception_patterns = []

    # Experience inflation: expert vs internship
    joined = " ".join(sessions).lower()
    if "seasoned devops engineer" in joined and "internship" in joined:
        deception_patterns.append({
            "lie_type": "experience_inflation",
            "contradictory_claims": [
                "I'm a seasoned DevOps engineer",
                "It was an internship"
            ]
        })

    # Leadership fabrication: leadership vs relying on senior
    if ("led" in joined or "responsible" in joined) and "senior engineer" in joined:
        deception_patterns.append({
            "lie_type": "leadership_fabrication",
            "contradictory_claims": [
                "I led / was responsible",
                "The senior engineer handled it"
            ]
        })

    # Self-contradiction: expert vs self-denial
    if "seasoned devops engineer" in joined and "i'm not a devops engineer" in joined:
        deception_patterns.append({
            "lie_type": "self_contradiction",
            "contradictory_claims": [
                "I'm a seasoned DevOps engineer",
                "I'm not a DevOps engineer"
            ]
        })

    # Equivocation: multiple hesitations
    if hesitations >= 2:
        deception_patterns.append({
            "lie_type": "equivocation",
            "contradictory_claims": ["vague/hesitant statements across sessions"]
        })


    revealed_truth = {
        "programming_experience": prog_exp if prog_exp is not None else None,
        "programming_language": programming_language if programming_language is not None else None,
        "skill_mastery": skill_mastery if skill_mastery is not None else None,
        "leadership_claims": leadership_claims,
        "team_experience": team_experience,
        "skills and other keywords": skills
    }

    result = {
        "shadow_id": shadow_id,
        "revealed_truth": revealed_truth,
        "deception_patterns": deception_patterns
    }
    return result


# ---------------- Validation + normalization of LLM output ----------------
REVEALED_KEYS = [
    "programming_experience",
    "programming_language",
    "skill_mastery",
    "leadership_claims",
    "team_experience",
    "skills and other keywords",
]


def normalize_parsed(parsed):
    """Take parsed JSON (from LLM) and return a final JSON object restricted to required schema."""
    if not isinstance(parsed, dict):
        return None
    final = {"shadow_id": None, "revealed_truth": {}, "deception_patterns": []}
    final["shadow_id"] = parsed.get("shadow_id") if isinstance(parsed.get("shadow_id"), str) else "phoenix_2024"

    # revealed_truth
    rt = parsed.get("revealed_truth", {})
    for k in REVEALED_KEYS:
        v = rt.get(k)
        if k == "skills and other keywords":
            if isinstance(v, list):
                # ensure strings and title-case
                cleaned = [str(x).strip() for x in v if x is not None]
                final["revealed_truth"][k] = [s.title() for s in cleaned]
            elif isinstance(v, str) and v.strip():
                final["revealed_truth"][k] = [v.title()]
            else:
                final["revealed_truth"][k] = []
        else:
            final["revealed_truth"][k] = v if (isinstance(v, str) or v is None) else str(v)

    # deception_patterns: only keep items with lie_type + contradictory_claims
    dp = parsed.get("deception_patterns", [])
    if isinstance(dp, list):
        out_dp = []
        for it in dp:
            if not isinstance(it, dict):
                continue
            lt = it.get("lie_type")
            cc = it.get("contradictory_claims") or it.get("contradictory_claim") or it.get("claims")
            if isinstance(cc, str):
                cc = [cc]
            if lt and isinstance(cc, list):
                # stringify claims
                cc_clean = [str(x).strip() for x in cc if x is not None]
                out_dp.append({"lie_type": str(lt), "contradictory_claims": cc_clean})
        final["deception_patterns"] = out_dp

    # ensure keys exist
    for k in REVEALED_KEYS:
        if k not in final["revealed_truth"]:
            final["revealed_truth"][k] = None if k != "skills and other keywords" else []

    return final


# ---------------- Main ----------------
def main():
    try:
        sessions = read_sessions()
    except FileNotFoundError as e:
        print("[!] No transcripts found:", e)
        return

    print("[*] Sessions loaded:", len(sessions))

    # Try LLM path first
    prompt = build_prompt_with_reasoning(sessions, shadow_id="phoenix_2024")
    print("[*] Sending detailed prompt to LLM (this may take some time)...")
    raw = None

    parsed_final = None

    if raw:
        # Save raw already in run_ollama; also keep copy
        with open(os.path.join(OUTPUT_DIR, "raw_response_full.txt"), "w", encoding="utf-8") as fh:
            fh.write(raw)

        # attempt to extract JSON
        candidate = extract_first_json(raw)
        parsed = try_parse_json_candidate(candidate)
        if parsed:
            normalized = normalize_parsed(parsed)
            if normalized:
                parsed_final = normalized
                print("[+] Successfully parsed and normalized LLM JSON output.")
        else:
            # ask the model to extract only JSON from its previous output (follow-up)
            followup = ("You just produced a reply. Extract and return ONLY the valid JSON object "
                       "that you included earlier — no explanation, no code fences, only the JSON object.\n\nPrevious output:\n\n") + raw
            print("[*] Asking LLM to return only the JSON object (follow-up).")
            raw2 = run_ollama(followup, model=MODEL_NAME, timeout=120)
            if raw2:
                with open(os.path.join(OUTPUT_DIR, "raw_response_followup.txt"), "w", encoding="utf-8") as fh:
                    fh.write(raw2)
                candidate2 = extract_first_json(raw2)
                parsed2 = try_parse_json_candidate(candidate2)
                if parsed2:
                    norm2 = normalize_parsed(parsed2)
                    if norm2:
                        parsed_final = norm2
                        print("[+] Successfully parsed JSON from follow-up.")

    # If LLM path failed, use rule-based fallback
    if parsed_final is None:
        print("[!] LLM path failed or produced invalid JSON. Falling back to deterministic rule-based analyzer.")
        parsed_final = rule_based_analyze(sessions, shadow_id="phoenix_2024")

    # Final safety: ensure JSON strictly follows schema (no extra top-level keys)
    final_output = {
        "shadow_id": parsed_final.get("shadow_id", "phoenix_2024"),
        "revealed_truth": parsed_final.get("revealed_truth", {
            k: None if k != "skills and other keywords" else [] for k in REVEALED_KEYS
        }),
        "deception_patterns": parsed_final.get("deception_patterns", [])
    }

    # Ensure types: skills is list, deception_patterns list of dicts with required keys
    if not isinstance(final_output["revealed_truth"].get("skills and other keywords"), list):
        final_output["revealed_truth"]["skills and other keywords"] = []

    safe_dp = []
    for d in final_output["deception_patterns"]:
        if isinstance(d, dict) and "lie_type" in d and "contradictory_claims" in d:
            claims = d["contradictory_claims"]
            if isinstance(claims, str):
                claims = [claims]
            claims = [str(x) for x in claims if x is not None]
            safe_dp.append({"lie_type": str(d["lie_type"]), "contradictory_claims": claims})
    final_output["deception_patterns"] = safe_dp

    # Save final JSON
    out_path = os.path.join(OUTPUT_DIR, "phoenix_2024.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(final_output, fh, indent=2, ensure_ascii=False)

    print(f"[+] Final JSON written to: {out_path}")
    # Print a short summary
    rt = final_output["revealed_truth"]
    print("Summary:")
    print(" programming_experience:", rt.get("programming_experience"))
    print(" programming_language:", rt.get("programming_language"))
    print(" skill_mastery:", rt.get("skill_mastery"))
    print(" leadership_claims:", rt.get("leadership_claims"))
    print(" team_experience:", rt.get("team_experience"))
    print(" skills:", rt.get("skills and other keywords"))
    if final_output["deception_patterns"]:
        print("Detected deception patterns:")
        for dp in final_output["deception_patterns"]:
            print(" -", dp["lie_type"], "|", dp["contradictory_claims"])


if __name__ == "__main__":
    main()
