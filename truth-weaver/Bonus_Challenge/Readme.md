## Bonus Challenge – Agentic Flow

The **Agentic Flow** models an **adaptive AI interviewer**.  

### Run the following command to get the output
```bash
cd Bonus_Challenge
python run_agent_demo.py
```

### States
- `idle_listening`
- `active_prompting`
- `deep_probe`
- `supportive`
- `clarification`
- `code_review`
- `wrap_up`

### Signals
- `silence_detected`
- `hesitation`
- `contradiction`
- `emotion_sobbing`
- `emotion_whispering`
- `overconfidence`
- `code_error`
- `time_up`

### Actions
- Stay silent, encourage, challenge, simplify, or probe deeper depending on state.  
- Wrap-up generates a **tailored summary** based on contradictions, overconfidence, or emotional stress.

### Deliverables
- `agent_flow.yaml` – structured transitions  
- `flow_diagram.png` – visual diagram  
- `agent_flow.py` – FSM logic  
- `run_agent_demo.py` – runs transcripts through agent, logs actions  
- `agentic_log.json` – full session timeline with states, actions, and final summary

---