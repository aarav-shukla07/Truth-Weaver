#!/usr/bin/env python3
# src/analyze.py
"""
Enhanced, robust transcript analyzer for Truth Weaver hackathon submission.

Key Improvements:
- Better error handling and input validation
- More sophisticated NLP pattern matching
- Enhanced deception detection algorithms
- Improved data structures and type hints
- Better logging and debugging capabilities
- More accurate skill and experience extraction
- Contextual analysis for better accuracy

ADDED FEATURE:
- Lightweight Ollama (llama3.1) fallback: only invoked when deterministic
  extraction misses important fields (programming_experience, programming_language,
  skills, etc.). If Ollama is not available or times out, deterministic result is used.
"""
import os
import re
import json
import math
import logging
import subprocess
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.absolute()
TRANSCRIPT_DIR = BASE_DIR.parent / "transcripts"
OUTPUT_DIR = BASE_DIR.parent / "../truth-weaver"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
SHADOW_ID = "PrelimsSubmission"
MAX_SESSIONS = 10

# Ollama / LLM fallback configuration
OLLAMA_ENABLED = True             # Set to False to disable Ollama fallback
OLLAMA_MODEL = "llama3.1"         # Local model name to call with `ollama run`
OLLAMA_TIMEOUT = 90               # seconds for subprocess call


@dataclass
class ExperienceClaim:
    """Structured representation of an experience claim"""
    raw_text: str
    numeric_value: Optional[float]
    session_id: int
    confidence: float = 1.0
    context: str = ""


@dataclass
class SessionAnalysis:
    """Complete analysis of a single session"""
    session_id: int
    text: str
    experience_claims: List[ExperienceClaim] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    mastery_level: Optional[str] = None
    leadership_indicators: List[str] = field(default_factory=list)
    team_indicators: List[str] = field(default_factory=list)
    hesitation_count: int = 0
    emotional_markers: List[str] = field(default_factory=list)
    credibility_score: float = 1.0


class TranscriptProcessor:
    """Enhanced transcript processing with better parsing capabilities"""
    
    # Expanded word-to-number mapping
    NUM_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
        "thirty": 30, "forty": 40, "fifty": 50
    }
    
    # Enhanced programming languages with variations
    PROGRAMMING_LANGUAGES = {
        "python": ["python", "py", "python3"],
        "java": ["java", "jvm"],
        "javascript": ["javascript", "js", "node", "nodejs", "react", "vue", "angular"],
        "typescript": ["typescript", "ts"],
        "c++": ["c++", "cpp", "cplus"],
        "c#": ["c#", "csharp", "c sharp"],
        "c": ["c language", " c "],
        "go": ["golang", "go lang"],
        "rust": ["rust"],
        "ruby": ["ruby", "rails"],
        "php": ["php"],
        "scala": ["scala"],
        "sql": ["sql", "mysql", "postgresql", "postgres"],
        "r": [" r ", "r language"],
        "swift": ["swift"],
        "kotlin": ["kotlin"],
        "dart": ["dart", "flutter"]
    }
    
    # Enhanced technical skills
    TECHNICAL_SKILLS = {
        "containerization": ["docker", "kubernetes", "k8s", "containerd", "podman"],
        "cloud_platforms": ["aws", "azure", "gcp", "google cloud", "amazon web services"],
        "infrastructure": ["terraform", "ansible", "puppet", "chef", "cloudformation"],
        "monitoring": ["prometheus", "grafana", "elk", "datadog", "newrelic"],
        "networking": ["calico", "istio", "envoy", "nginx", "haproxy", "dns", "tcp/ip"],
        "ci_cd": ["jenkins", "gitlab", "github actions", "circleci", "travis"],
        "databases": ["mongodb", "redis", "elasticsearch", "cassandra", "dynamodb"],
        "ml_ai": ["machine learning", "ml", "tensorflow", "pytorch", "scikit-learn"],
        "web_frameworks": ["django", "flask", "spring", "express", "laravel"],
        "version_control": ["git", "github", "gitlab", "bitbucket", "svn"]
    }
    
    @staticmethod
    def load_sessions(folder: Path = TRANSCRIPT_DIR, max_sessions: int = MAX_SESSIONS) -> List[str]:
        """
        Enhanced session loading with support for audio-filename-based transcript splits.
        """
        if not folder.exists():
            raise FileNotFoundError(f"Transcripts folder not found: {folder}")

        logger.info(f"Loading sessions from {folder}")

        # Prioritize our combined transcribed.txt
        file_path = folder / "transcribed.txt"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Split by audio filename markers: === FILENAME ===
                parts = re.split(r"={3,}\s*([^\n=]+)\s*={3,}", content)
                # re.split keeps delimiters in the result. We want pairs (filename, transcript)
                sessions = []
                for i in range(1, len(parts), 2):
                    filename = parts[i].strip()
                    transcript = parts[i+1].strip()
                    if transcript:
                        # store with filename marker at start for identification
                        sessions.append(f"[{filename}] {transcript}")

                if sessions:
                    logger.info(f"Split transcript into {len(sessions)} sessions (by filenames)")
                    return sessions[:max_sessions]

            except Exception as e:
                logger.warning(f"Failed to process transcribed.txt: {e}")

        # fallback to old handling (in case someone provides session1.txt etc.)
        txt_files = list(folder.glob("*.txt"))
        if txt_files:
            sessions = []
            for file_path in sorted(txt_files)[:max_sessions]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        sessions.append(content)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            return sessions

        logger.warning("No transcript files found")
        return []


class ExperienceExtractor:
    """Enhanced experience extraction with better pattern matching"""
    
    def __init__(self, processor: TranscriptProcessor):
        self.processor = processor
    
    def extract_experience_claims(self, text: str, session_id: int) -> List[ExperienceClaim]:
        """
        Extract experience claims with enhanced pattern matching and context awareness
        """
        claims = []
        text_lower = text.lower()
        
        # Pattern 1: Direct numeric years
        for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(years?|yrs?)\s*(?:of\s+)?(?:experience|work|programming)?', text_lower):
            years = float(match.group(1))
            context = self._extract_context(text, match.start(), match.end())
            claims.append(ExperienceClaim(
                raw_text=match.group(0),
                numeric_value=years,
                session_id=session_id,
                confidence=0.9,
                context=context
            ))
        
        # Pattern 2: Months converted to years
        for match in re.finditer(r'(\d+(?:\.\d+)?)\s*months?', text_lower):
            months = float(match.group(1))
            context = self._extract_context(text, match.start(), match.end())
            claims.append(ExperienceClaim(
                raw_text=match.group(0),
                numeric_value=months / 12.0,
                session_id=session_id,
                confidence=0.8,
                context=context
            ))
        
        # Pattern 3: Spelled-out numbers
        for word, value in self.processor.NUM_WORDS.items():
            pattern = rf'\b{word}\s+years?\b'
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                context = self._extract_context(text, match.start(), match.end())
                claims.append(ExperienceClaim(
                    raw_text=f"{word} years",
                    numeric_value=float(value),
                    session_id=session_id,
                    confidence=0.7,
                    context=context
                ))
        
        # Pattern 4: Relative time expressions
        relative_patterns = {
            r'almost\s+(\d+)\s+years?': lambda x: float(x) - 0.5,
            r'about\s+(\d+)\s+years?': lambda x: float(x),
            r'around\s+(\d+)\s+years?': lambda x: float(x),
            r'over\s+(\d+)\s+years?': lambda x: float(x) + 1.0,
            r'more than\s+(\d+)\s+years?': lambda x: float(x) + 1.0,
            r'less than\s+(\d+)\s+years?': lambda x: max(0.5, float(x) - 1.0),
            r'under\s+(\d+)\s+years?': lambda x: max(0.5, float(x) - 1.0)
        }
        
        for pattern, value_func in relative_patterns.items():
            for match in re.finditer(pattern, text_lower):
                try:
                    years = value_func(match.group(1))
                    context = self._extract_context(text, match.start(), match.end())
                    claims.append(ExperienceClaim(
                        raw_text=match.group(0),
                        numeric_value=years,
                        session_id=session_id,
                        confidence=0.6,
                        context=context
                    ))
                except (ValueError, IndexError):
                    continue
        
        # Pattern 5: Vague expressions (no numeric value)
        vague_patterns = [
            "couple of years", "few years", "several years", "many years",
            "long time", "extensive experience", "years of experience",
            "plenty of experience", "significant experience"
        ]
        
        for pattern in vague_patterns:
            if pattern in text_lower:
                match = re.search(re.escape(pattern), text_lower)
                context = self._extract_context(text, match.start(), match.end())
                claims.append(ExperienceClaim(
                    raw_text=pattern,
                    numeric_value=None,
                    session_id=session_id,
                    confidence=0.3,
                    context=context
                ))
        
        # Pattern 6: Career level indicators
        career_indicators = {
            "intern": 0.2, "internship": 0.2, "entry level": 0.5,
            "junior": 1.0, "mid-level": 3.0, "senior": 7.0,
            "lead": 8.0, "principal": 10.0, "architect": 12.0
        }
        
        for indicator, years in career_indicators.items():
            if indicator in text_lower:
                match = re.search(re.escape(indicator), text_lower)
                context = self._extract_context(text, match.start(), match.end())
                claims.append(ExperienceClaim(
                    raw_text=indicator,
                    numeric_value=years,
                    session_id=session_id,
                    confidence=0.4,
                    context=context
                ))
        
        return claims
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract surrounding context for better analysis"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()


class SkillExtractor:
    """Enhanced skill extraction with contextual analysis"""
    
    def __init__(self, processor: TranscriptProcessor):
        self.processor = processor
    
    def extract_languages_and_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract programming languages and technical skills with confidence scoring"""
        text_lower = text.lower()
        languages = []
        skills = []
        
        # Extract programming languages
        for lang, variations in self.processor.PROGRAMMING_LANGUAGES.items():
            for variation in variations:
                # Look for contextual mentions
                patterns = [
                    rf'\b(?:experience|work|code|program|develop|use|using|with|in)\s+(?:in\s+)?{re.escape(variation)}\b',
                    rf'\b{re.escape(variation)}\s+(?:programming|development|coding|experience)\b',
                    rf'\b{re.escape(variation)}\b'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        if lang not in languages:
                            languages.append(lang)
                        break
        
        # Extract technical skills
        for skill_category, keywords in self.processor.TECHNICAL_SKILLS.items():
            for keyword in keywords:
                patterns = [
                    rf'\b(?:experience|work|use|using|with|manage|deploy)\s+(?:with\s+)?{re.escape(keyword)}\b',
                    rf'\b{re.escape(keyword)}\s+(?:experience|deployment|management)\b',
                    rf'\b{re.escape(keyword)}\b'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        formatted_skill = self._format_skill_name(keyword)
                        if formatted_skill not in skills:
                            skills.append(formatted_skill)
                        break
        
        return languages, skills
    
    def _format_skill_name(self, skill: str) -> str:
        """Format skill names consistently"""
        # Special cases
        special_formats = {
            "k8s": "Kubernetes",
            "ml": "Machine Learning",
            "ai": "Artificial Intelligence",
            "ci_cd": "CI/CD",
            "aws": "AWS",
            "gcp": "Google Cloud Platform",
            "tcp/ip": "TCP/IP"
        }
        
        if skill.lower() in special_formats:
            return special_formats[skill.lower()]
        
        return skill.title()


class DeceptionDetector:
    """Enhanced deception detection with sophisticated pattern analysis"""
    
    def __init__(self):
        self.contradiction_patterns = []
        self.inconsistency_threshold = 0.7
    
    def detect_deception_patterns(self, sessions: List[SessionAnalysis]) -> List[Dict[str, Any]]:
        """
        Detect various deception patterns with improved accuracy
        """
        patterns = []
        
        # 1. Experience inconsistencies
        exp_pattern = self._detect_experience_inconsistencies(sessions)
        if exp_pattern:
            patterns.append(exp_pattern)
        
        # 2. Leadership fabrication
        leadership_pattern = self._detect_leadership_fabrication(sessions)
        if leadership_pattern:
            patterns.append(leadership_pattern)
        
        # 3. Team experience contradictions
        team_pattern = self._detect_team_contradictions(sessions)
        if team_pattern:
            patterns.append(team_pattern)
        
        # 4. Skill inflation
        skill_pattern = self._detect_skill_inflation(sessions)
        if skill_pattern:
            patterns.append(skill_pattern)
        
        # 5. Emotional inconsistency
        emotional_pattern = self._detect_emotional_inconsistency(sessions)
        if emotional_pattern:
            patterns.append(emotional_pattern)
        
        # 6. Vagueness as evasion
        vague_pattern = self._detect_vagueness_pattern(sessions)
        if vague_pattern:
            patterns.append(vague_pattern)
        
        return patterns
    
    def _detect_experience_inconsistencies(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect inconsistencies in experience claims"""
        all_claims = []
        for session in sessions:
            all_claims.extend(session.experience_claims)
        
        # Filter numeric claims
        numeric_claims = [claim for claim in all_claims if claim.numeric_value is not None]
        
        if len(numeric_claims) < 2:
            return None
        
        # Check for significant variations
        values = [claim.numeric_value for claim in numeric_claims]
        min_exp, max_exp = min(values), max(values)
        
        # Consider claims inconsistent if they vary by more than 2 years
        if max_exp - min_exp > 2.0:
            contradictory_claims = []
            for claim in numeric_claims:
                if claim.numeric_value == min_exp or claim.numeric_value == max_exp:
                    contradictory_claims.append(f"Session {claim.session_id}: {claim.raw_text}")
            
            return {
                "lie_type": "experience_inconsistency",
                "contradictory_claims": contradictory_claims,
            }
        
        return None
    
    def _detect_leadership_fabrication(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect fabricated leadership claims"""
        leadership_sessions = []
        passive_sessions = []
        
        for session in sessions:
            if session.leadership_indicators:
                leadership_sessions.append(session)
            
            # Check for passive language
            passive_indicators = ["watched", "observed", "followed", "assisted", "helped"]
            if any(indicator in session.text.lower() for indicator in passive_indicators):
                passive_sessions.append(session)
        
        if leadership_sessions and passive_sessions:
            claims = []
            for session in leadership_sessions:
                claims.append(f"Session {session.session_id}: Leadership claims")
            for session in passive_sessions:
                claims.append(f"Session {session.session_id}: Passive role admission")
            
            return {
                "lie_type": "leadership_fabrication",
                "contradictory_claims": claims,
            }
        
        return None
    
    def _detect_team_contradictions(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect team experience contradictions"""
        team_sessions = []
        individual_sessions = []
        
        for session in sessions:
            team_words = ["team", "we", "our", "collaborate", "group"]
            individual_words = ["alone", "myself", "independently", "solo"]
            
            if any(word in session.text.lower() for word in team_words):
                team_sessions.append(session)
            if any(word in session.text.lower() for word in individual_words):
                individual_sessions.append(session)
        
        if team_sessions and individual_sessions:
            claims = []
            claims.extend([f"Session {s.session_id}: Team collaboration mentioned" for s in team_sessions])
            claims.extend([f"Session {s.session_id}: Individual work mentioned" for s in individual_sessions])
            
            return {
                "lie_type": "team_experience_contradiction",
                "contradictory_claims": claims,
            }
        
        return None
    
    def _detect_skill_inflation(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect skill inflation patterns"""
        skill_progression = {}
        
        for session in sessions:
            for skill in session.skills:
                if skill not in skill_progression:
                    skill_progression[skill] = []
                skill_progression[skill].append(session.session_id)
        
        # Look for skills mentioned in later sessions but not earlier ones (potential inflation)
        inflated_skills = []
        for skill, session_ids in skill_progression.items():
            if len(session_ids) == 1 and session_ids[0] > 1:
                inflated_skills.append(skill)
        
        if len(inflated_skills) > 2:  # Multiple new skills appearing late
            return {
                "lie_type": "skill_inflation",
                "contradictory_claims": [f"Late introduction of skills: {', '.join(inflated_skills)}"],
            }
        
        return None
    
    def _detect_emotional_inconsistency(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect emotional inconsistencies that may indicate deception"""
        early_confidence = any(len(s.emotional_markers) == 0 and len(s.leadership_indicators) > 0 
                              for s in sessions[:2])
        later_emotional = any(len(s.emotional_markers) > 0 for s in sessions[2:])
        
        if early_confidence and later_emotional:
            return {
                "lie_type": "emotional_inconsistency",
                "contradictory_claims": [
                    "Early confident claims",
                    "Later emotional distress or uncertainty"
                ],
            }
        
        return None
    
    def _detect_vagueness_pattern(self, sessions: List[SessionAnalysis]) -> Optional[Dict[str, Any]]:
        """Detect patterns of vagueness as potential evasion"""
        vague_claims = []
        for session in sessions:
            vague_in_session = [claim for claim in session.experience_claims 
                               if claim.numeric_value is None]
            if len(vague_in_session) > 1:
                vague_claims.extend([f"Session {session.session_id}: {claim.raw_text}" 
                                   for claim in vague_in_session])
        
        if len(vague_claims) > 3:
            return {
                "lie_type": "evasive_vagueness",
                "contradictory_claims": vague_claims,
            }
        
        return None


class TruthAnalyzer:
    """Main analyzer class that coordinates all components"""
    
    def __init__(self):
        self.processor = TranscriptProcessor()
        self.experience_extractor = ExperienceExtractor(self.processor)
        self.skill_extractor = SkillExtractor(self.processor)
        self.deception_detector = DeceptionDetector()
    
    # ---------- LLM / Ollama fallback helpers ----------
    @staticmethod
    def _extract_json_balanced(text: str) -> Optional[str]:
        """
        Extract the first balanced {...} JSON substring from a blob of text.
        Handles quoted strings and escapes.
        """
        if not text:
            return None
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
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
                        return text[start:i+1]
        return None

    def _ask_ollama_fill(self, sessions: List[str], missing_fields: List[str]) -> Dict[str, Any]:
        """
        Ask local Ollama (llama3.1) to fill missing fields.
        Returns parsed JSON dict with keys matching missing_fields (if available).
        Robust to timeouts/errors: returns {} on failure.
        """
        if not OLLAMA_ENABLED:
            logger.info("[+] Ollama fallback disabled by configuration.")
            return {}

        # Build compact prompt that asks to return only a JSON object with the missing keys.
        missing_json_schema = {k: "null" for k in missing_fields}
        prompt = (
            "You are an assistant that extracts interview metadata. "
            "Return ONLY one valid JSON object (no explanation) containing these keys exactly:\n"
            f"{json.dumps(list(missing_fields))}\n\n"
            "Use the transcripts below. Be conservative: if unsure, return null or empty list.\n\n"
            "Transcripts:\n"
        )
        for i, s in enumerate(sessions, start=1):
            snippet = s.strip().replace("\n", " ")[:1000]
            prompt += f"\nSession {i}: {snippet}"

        prompt += (
            "\n\nOutput JSON should only include the requested keys and basic primitive values "
            "(strings or lists). Example: {\"programming_experience\":\"3-4 years\",\"skills and other keywords\":[\"Kubernetes\"]}"
        )

        try:
            logger.info("[*] Calling Ollama for missing fields...")
            proc = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=OLLAMA_TIMEOUT
            )
            stdout = proc.stdout.decode("utf-8", errors="ignore")
            stderr = proc.stderr.decode("utf-8", errors="ignore")
            if stderr:
                logger.debug(f"Ollama stderr: {stderr[:1000]}")
            candidate = self._extract_json_balanced(stdout)
            if not candidate:
                # try fallback: maybe the model printed JSON-like lines
                candidate = self._extract_json_balanced(stderr)
            if not candidate:
                logger.warning("[!] Ollama did not return a JSON object (or parsing failed).")
                return {}
            try:
                parsed = json.loads(candidate)
                logger.info("[+] Ollama returned JSON; merging results for missing fields.")
                return parsed if isinstance(parsed, dict) else {}
            except Exception as e:
                logger.warning(f"[!] Failed to parse JSON from Ollama output: {e}")
                return {}
        except subprocess.TimeoutExpired:
            logger.warning("[!] Ollama call timed out.")
            return {}
        except FileNotFoundError:
            logger.warning("[!] Ollama CLI not found (ensure `ollama` is installed and on PATH).")
            return {}
        except Exception as e:
            logger.warning(f"[!] Ollama call failed: {e}")
            return {}

    # ---------- analysis pipeline ----------
    def analyze_sessions(self, sessions: List[str]) -> List[SessionAnalysis]:
        """Analyze all sessions and return structured results"""
        analyses = []
        
        for i, session_text in enumerate(sessions, 1):
            logger.info(f"Analyzing session {i}")
            
            analysis = SessionAnalysis(
                session_id=i,
                text=session_text
            )
            
            # Extract experience claims
            analysis.experience_claims = self.experience_extractor.extract_experience_claims(
                session_text, i
            )
            
            # Extract languages and skills
            analysis.languages, analysis.skills = self.skill_extractor.extract_languages_and_skills(
                session_text
            )
            
            # Detect mastery level
            analysis.mastery_level = self._detect_mastery_level(session_text)
            
            # Detect leadership indicators
            analysis.leadership_indicators = self._detect_leadership_indicators(session_text)
            
            # Detect team indicators
            analysis.team_indicators = self._detect_team_indicators(session_text)
            
            # Count hesitations and emotional markers
            analysis.hesitation_count = self._count_hesitations(session_text)
            analysis.emotional_markers = self._detect_emotional_markers(session_text)
            
            # Calculate credibility score
            analysis.credibility_score = self._calculate_credibility_score(analysis)
            
            analyses.append(analysis)
            
            logger.info(f"Session {i} analysis complete: "
                       f"{len(analysis.experience_claims)} experience claims, "
                       f"{len(analysis.skills)} skills detected")
        
        return analyses
    
    def _detect_mastery_level(self, text: str) -> Optional[str]:
        """Detect skill mastery level from text"""
        text_lower = text.lower()
        
        advanced_indicators = ["expert", "master", "senior", "lead", "architect", 
                              "extensive experience", "deep knowledge"]
        beginner_indicators = ["beginner", "junior", "learning", "new to", 
                              "getting started", "intern", "entry level"]
        
        if any(indicator in text_lower for indicator in advanced_indicators):
            return "advanced"
        elif any(indicator in text_lower for indicator in beginner_indicators):
            return "beginner"
        else:
            # Default to intermediate if skills are mentioned but no explicit level
            # If no skills at all, leave None to allow Ollama fallback
            langs, skills = self.skill_extractor.extract_languages_and_skills(text)
            if skills:
                return "intermediate"
            return None
    
    def _detect_leadership_indicators(self, text: str) -> List[str]:
        """Detect leadership indicators in text"""
        text_lower = text.lower()
        indicators = []
        
        leadership_patterns = [
            "led", "leading", "managed", "supervised", "coordinated",
            "responsible for", "owned", "headed", "directed", "guided"
        ]
        
        for pattern in leadership_patterns:
            if pattern in text_lower:
                indicators.append(pattern)
        
        return indicators
    
    def _detect_team_indicators(self, text: str) -> List[str]:
        """Detect team collaboration indicators"""
        text_lower = text.lower()
        indicators = []
        
        team_patterns = ["team", "we", "our", "collaborate", "together", 
                        "group", "pair programming", "worked with"]
        
        for pattern in team_patterns:
            if pattern in text_lower:
                indicators.append(pattern)
        
        return indicators
    
    def _count_hesitations(self, text: str) -> int:
        """Count hesitation markers"""
        hesitation_patterns = ["uh", "um", "maybe", "probably", "i think", 
                              "i guess", "kind of", "sort of", "not sure"]
        text_lower = text.lower()
        
        count = 0
        for pattern in hesitation_patterns:
            count += len(re.findall(rf'\b{re.escape(pattern)}\b', text_lower))
        
        return count
    
    def _detect_emotional_markers(self, text: str) -> List[str]:
        """Detect emotional markers in text"""
        text_lower = text.lower()
        markers = []
        
        emotional_patterns = ["crying", "sobbing", "sigh", "nervous", 
                            "anxious", "worried", "stressed", "upset"]
        
        for pattern in emotional_patterns:
            if pattern in text_lower:
                markers.append(pattern)
        
        return markers
    
    def _calculate_credibility_score(self, analysis: SessionAnalysis) -> float:
        """Calculate a credibility score for the session"""
        score = 1.0
        
        # Reduce score for high hesitation
        if analysis.hesitation_count > 5:
            score -= 0.2
        
        # Reduce score for emotional distress
        if analysis.emotional_markers:
            score -= 0.1 * len(analysis.emotional_markers)
        
        # Reduce score for vague experience claims
        vague_claims = [claim for claim in analysis.experience_claims 
                       if claim.numeric_value is None]
        if len(vague_claims) > 2:
            score -= 0.3
        
        return max(0.0, score)
    
    def synthesize_results(self, analyses: List[SessionAnalysis], sessions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synthesize final results from all session analyses

        NOTE: accepts `sessions` (raw text list) so that Ollama fallback can use them when needed.
        """
        
        # Aggregate experience
        all_experience_claims = []
        for analysis in analyses:
            all_experience_claims.extend(analysis.experience_claims)
        
        programming_experience = self._determine_programming_experience(all_experience_claims)
        
        # Aggregate languages
        language_counter = Counter()
        for analysis in analyses:
            for lang in analysis.languages:
                language_counter[lang] += 1
        
        programming_language = language_counter.most_common(1)[0][0] if language_counter else None
        
        # Aggregate skills
        all_skills = []
        for analysis in analyses:
            all_skills.extend(analysis.skills)
        
        unique_skills = list(dict.fromkeys(all_skills))  # Preserve order, remove duplicates
        
        # Determine mastery
        mastery_votes = [a.mastery_level for a in analyses if a.mastery_level]
        skill_mastery = Counter(mastery_votes).most_common(1)[0][0] if mastery_votes else None
        
        # Determine leadership
        leadership_claims = self._determine_leadership_claims(analyses)
        
        # Determine team experience
        team_experience = self._determine_team_experience(analyses)
        
        # Detect deception patterns
        deception_patterns = self.deception_detector.detect_deception_patterns(analyses)

        # Build results dict (internal keys)
        results = {
            "programming_experience": programming_experience,
            "programming_language": programming_language,
            "skill_mastery": skill_mastery,
            "leadership_claims": leadership_claims,
            "team_experience": team_experience,
            "skills_and_keywords": unique_skills,
            "deception_patterns": deception_patterns
        }

        # ------------------ Ollama fallback (only if missing important fields) ------------------
        # Determine which fields are missing or low-confidence
        missing = []
        # consider "programming_experience", "programming_language", "skills_and_keywords", "skill_mastery"
        if results["programming_experience"] in (None, ""):
            missing.append("programming_experience")
        if results["programming_language"] in (None, "", []):
            missing.append("programming_language")
        if not results["skills_and_keywords"]:
            missing.append("skills and other keywords")  # use final schema name so Ollama returns correct key
        if results["skill_mastery"] in (None, "", "unclear"):
            missing.append("skill_mastery")

        if missing and sessions:
            # ask Ollama to fill only missing fields
            ai_response = self._ask_ollama_fill(sessions, missing)
            if isinstance(ai_response, dict) and ai_response:
                # Map AI keys into our internal results keys
                # Accept multiple possible key forms
                for k in missing:
                    # direct key
                    if k in ai_response and ai_response[k]:
                        val = ai_response[k]
                    # maybe AI returns "skills_and_keywords" or "skills"
                    elif k == "skills and other keywords" and ("skills_and_keywords" in ai_response or "skills" in ai_response):
                        val = ai_response.get("skills_and_keywords") or ai_response.get("skills")
                    else:
                        val = ai_response.get(k) if isinstance(k, str) else None

                    # Normalize and set if meaningful
                    if val:
                        if k in ("skills and other keywords", "skills_and_keywords", "skills"):
                            # ensure list
                            if isinstance(val, str):
                                # try comma-separated
                                parts = [s.strip() for s in re.split(r",|\n|;", val) if s.strip()]
                                results["skills_and_keywords"] = parts
                            elif isinstance(val, list):
                                results["skills_and_keywords"] = val
                        else:
                            # simple assignment for strings
                            if isinstance(val, list):
                                # pick first element if list was returned
                                results[k] = val[0] if val else results.get(k)
                            else:
                                results[k] = val

        # Final normalization for schema users expect
        if not isinstance(results["skills_and_keywords"], list):
            results["skills_and_keywords"] = list(results["skills_and_keywords"]) if results["skills_and_keywords"] else []

        return results
    
    def _determine_programming_experience(self, claims: List[ExperienceClaim]) -> Optional[str]:
        """Determine programming experience from all claims"""
        if not claims:
            return None
        
        # Filter and weight claims by confidence
        numeric_claims = [(claim.numeric_value, claim.confidence) 
                         for claim in claims if claim.numeric_value is not None]
        
        if not numeric_claims:
            # Check for internship or entry-level indicators
            intern_claims = [claim for claim in claims if "intern" in claim.raw_text.lower()]
            if intern_claims:
                return "0-1 years"
            return None
        
        # Calculate weighted average
        total_weighted_value = sum(value * confidence for value, confidence in numeric_claims)
        total_weight = sum(confidence for _, confidence in numeric_claims)
        
        if total_weight == 0:
            return None
        
        avg_experience = total_weighted_value / total_weight
        
        # Round to meaningful ranges
        if avg_experience < 1.0:
            return "0-1 years"
        elif avg_experience < 2.0:
            return "1-2 years"
        elif avg_experience < 5.0:
            return f"{int(round(avg_experience))} years"
        else:
            return f"{int(round(avg_experience))}+ years"
    
    def _determine_leadership_claims(self, analyses: List[SessionAnalysis]) -> str:
        """Determine the nature of leadership claims"""
        leadership_sessions = [a for a in analyses if a.leadership_indicators]
        passive_sessions = [a for a in analyses if any(
            word in a.text.lower() for word in ["watched", "observed", "intern", "assisted"]
        )]
        
        if not leadership_sessions:
            return "none"
        
        if passive_sessions:
            # Leadership claims contradicted by passive admissions
            return "fabricated"
        
        # Check credibility scores
        avg_credibility = sum(a.credibility_score for a in leadership_sessions) / len(leadership_sessions)
        
        if avg_credibility > 0.7:
            return "genuine"
        else:
            return "questionable"
    
    def _determine_team_experience(self, analyses: List[SessionAnalysis]) -> str:
        """Determine team experience pattern"""
        team_sessions = [a for a in analyses if a.team_indicators]
        individual_indicators = []
        
        for analysis in analyses:
            if any(word in analysis.text.lower() for word in 
                  ["alone", "myself", "independently", "solo", "individual"]):
                individual_indicators.append(analysis)
        
        if team_sessions and not individual_indicators:
            return "team player"
        elif individual_indicators and not team_sessions:
            return "individual contributor"
        elif team_sessions and individual_indicators:
            return "mixed"
        else:
            return "unclear"
    
    def generate_report(self, sessions: List[str]) -> Dict[str, Any]:
        """Generate complete analysis report"""
        logger.info("Starting transcript analysis")
        
        # Analyze all sessions
        session_analyses = self.analyze_sessions(sessions)
        
        # Synthesize results (pass original sessions for LLM fallback)
        results = self.synthesize_results(session_analyses, sessions)
        
        # Format for required schema
        final_report = {
            "shadow_id": SHADOW_ID,
            "revealed_truth": {
                "programming_experience": results.get("programming_experience"),
                "programming_language": (results.get("programming_language") or None),
                "skill_mastery": results.get("skill_mastery"),
                "leadership_claims": results.get("leadership_claims"),
                "team_experience": results.get("team_experience"),
                "skills and other keywords": results.get("skills_and_keywords", [])
            },
            "deception_patterns": results.get("deception_patterns", [])
        }
        
        # Log summary
        logger.info("Analysis complete:")
        logger.info(f"  Programming experience: {results['programming_experience']}")
        logger.info(f"  Primary language: {results['programming_language']}")
        logger.info(f"  Skill mastery: {results['skill_mastery']}")
        logger.info(f"  Leadership claims: {results['leadership_claims']}")
        logger.info(f"  Team experience: {results['team_experience']}")
        logger.info(f"  Skills detected: {len(results['skills_and_keywords'])}")
        logger.info(f"  Deception patterns: {len(results['deception_patterns'])}")
        
        return final_report


class ReportGenerator:
    """Generate detailed analysis reports"""
    
    @staticmethod
    def save_report(report: Dict[str, Any], output_path: Path) -> None:
        """Save report to JSON file with proper formatting"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
    
    @staticmethod
    def print_summary(report: Dict[str, Any]) -> None:
        """Print a human-readable summary"""
        print("\n" + "="*60)
        print("TRANSCRIPT ANALYSIS SUMMARY")
        print("="*60)
        
        truth = report["revealed_truth"]
        
        print(f"\nCandidate ID: {report['shadow_id']}")
        print(f"Programming Experience: {truth.get('programming_experience', 'Unknown')}")
        print(f"Primary Language: {truth.get('programming_language', 'Unknown')}")
        print(f"Skill Mastery: {truth.get('skill_mastery', 'Unknown')}")
        print(f"Leadership Claims: {truth.get('leadership_claims', 'Unknown')}")
        print(f"Team Experience: {truth.get('team_experience', 'Unknown')}")
        
        skills = truth.get('skills and other keywords', [])
        if skills:
            print(f"\nTechnical Skills ({len(skills)}):")
            for skill in skills:
                print(f"  • {skill}")
        
        patterns = report.get("deception_patterns", [])
        if patterns:
            print(f"\nDeception Patterns Detected ({len(patterns)}):")
            for pattern in patterns:
                print(f"  • {pattern['lie_type'].replace('_', ' ').title()}")
                print(f"    Evidence: {len(pattern['contradictory_claims'])} contradictory claims")
        else:
            print("\nNo deception patterns detected.")
        
        print("\n" + "="*60)


def main():
    """Enhanced main function with better error handling"""
    try:
        # Initialize analyzer
        analyzer = TruthAnalyzer()
        
        # Load sessions
        sessions = TranscriptProcessor.load_sessions()
        
        if not sessions:
            logger.error("No transcript files found. Please ensure transcript files are in the transcripts/ directory.")
            logger.info("Expected file patterns:")
            logger.info("  - session1.txt, session2.txt, ...")
            logger.info("  - transcript.txt (with session markers)")
            logger.info("  - Any .txt files")
            return
        
        logger.info(f"Successfully loaded {len(sessions)} session(s)")
        
        # Generate analysis report
        report = analyzer.generate_report(sessions)
        
        # Save report
        output_path = OUTPUT_DIR / f"{SHADOW_ID}.json"
        ReportGenerator.save_report(report, output_path)
        
        # Print summary
        ReportGenerator.print_summary(report)
        
        # Validation check
        if not _validate_report_schema(report):
            logger.warning("Generated report may not match expected schema")
        else:
            logger.info("Report validation passed")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Please ensure the transcripts directory exists and contains transcript files")
    except json.JSONEncodeError as e:
        logger.error(f"JSON encoding error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        logger.exception("Full traceback:")


def _validate_report_schema(report: Dict[str, Any]) -> bool:
    """Validate that the report matches the expected schema"""
    try:
        # Check required top-level keys
        required_keys = {"shadow_id", "revealed_truth", "deception_patterns"}
        if not all(key in report for key in required_keys):
            return False
        
        # Check revealed_truth structure
        truth = report["revealed_truth"]
        truth_keys = {
            "programming_experience", "programming_language", "skill_mastery",
            "leadership_claims", "team_experience", "skills and other keywords"
        }
        if not all(key in truth for key in truth_keys):
            return False
        
        # Check that skills is a list
        if not isinstance(truth["skills and other keywords"], list):
            return False
        
        # Check deception_patterns structure
        patterns = report["deception_patterns"]
        if not isinstance(patterns, list):
            return False
        
        for pattern in patterns:
            if not isinstance(pattern, dict):
                return False
            if not all(key in pattern for key in ["lie_type", "contradictory_claims"]):
                return False
            if not isinstance(pattern["contradictory_claims"], list):
                return False
        
        return True
        
    except (KeyError, TypeError):
        return False


if __name__ == "__main__":
    main()
