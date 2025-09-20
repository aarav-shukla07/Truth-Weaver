class AgenticFlow:
    def __init__(self):
        self.state = "idle_listening"
        self.history = []   # store signals
        self.flags = {"contradictions": 0, "overconfidence": 0, "sobbing": 0}

    def handle_signal(self, signal):
        self.history.append((self.state, signal))
        
        # Update counters
        if signal == "contradiction":
            self.flags["contradictions"] += 1
        if signal == "overconfidence":
            self.flags["overconfidence"] += 1
        if signal == "emotion_sobbing":
            self.flags["sobbing"] += 1

        transitions = {
            # --- Idle Listening ---
            ("idle_listening", "silence_detected"): ("active_prompting", "Ask a gentle nudge question"),
            ("idle_listening", "contradiction"): ("deep_probe", "Highlight inconsistency in answers"),
            ("idle_listening", "emotion_sobbing"): ("supportive", "Encourage candidate and lower pressure"),
            ("idle_listening", "emotion_whispering"): ("supportive", "Ask them to speak louder, reassure"),
            ("idle_listening", "overconfidence"): ("deep_probe", "Challenge inflated claim with a harder question"),
            ("idle_listening", "hesitation"): ("clarification", "Ask candidate to elaborate with a concrete example"),

            # --- Deep Probe ---
            ("deep_probe", "contradiction"): ("deep_probe", "Press candidate until they resolve inconsistency"),
            ("deep_probe", "overconfidence"): ("deep_probe", "Demand technical depth to verify claim"),
            ("deep_probe", "hesitation"): ("clarification", "Simplify and redirect question"),
            ("deep_probe", "emotion_sobbing"): ("supportive", "Shift to reassurance, then probe again gently"),

            # --- Supportive ---
            ("supportive", "hesitation"): ("supportive", "Give candidate more time and reduce difficulty"),
            ("supportive", "contradiction"): ("deep_probe", "Revisit inconsistency in a gentler way"),
            ("supportive", "overconfidence"): ("deep_probe", "Balance support with challenge"),

            # --- Clarification ---
            ("clarification", "hesitation"): ("clarification", "Repeat in simpler words"),
            ("clarification", "contradiction"): ("deep_probe", "Clarify inconsistency after vague answer"),
            ("clarification", "emotion_sobbing"): ("supportive", "Soften tone: reassure before retrying question"),
            ("clarification", "overconfidence"): ("deep_probe", "Push candidate for technical detail"),
        }

        # --- Wrap-up always available ---
        if signal == "time_up":
            self.state = "wrap_up"
            return self.generate_summary()

        # Transition lookup
        key = (self.state, signal)
        if key in transitions:
            new_state, action = transitions[key]
            self.state = new_state
            return action
        else:
            return f"[Unhandled] In {self.state}, got {signal}"

    def generate_summary(self):
        """Summarize based on candidate behavior flags"""
        summary = "Final Interview Summary: "
        if self.flags["overconfidence"] > 0 and self.flags["contradictions"] > 0:
            summary += "Candidate made inflated claims but inconsistencies were revealed."
        elif self.flags["sobbing"] > 0:
            summary += "Candidate showed emotional stress, needing reassurance."
        elif self.flags["contradictions"] > 0:
            summary += "Candidate gave conflicting answers, truth is uncertain."
        elif self.flags["overconfidence"] > 0:
            summary += "Candidate was confident but may have overstated experience."
        else:
            summary += "Candidate was consistent and cooperative."
        return summary
