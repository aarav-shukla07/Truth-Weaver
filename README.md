# Truth-Weaver

## Overview
Truth Weaver is an intelligent system designed to analyze technical interviews.  
It processes **multi-session audio interviews**, extracts **truths vs. deceptions**, and models an **adaptive interviewer (Agentic Flow)**.  

The system has two main parts:
1. **Truth Analysis Pipeline** – Audio → Transcripts → Truth Extraction → Evaluation
2. **Agentic Flow (Bonus Challenge)** – A state-based AI interviewer that adapts to candidate signals.

---

## How to Run

1. Install Dependencies
   ```bash
   pip install -r requirements.txt
    ```
    Additional setup:
    - Ollama must be installed.
    - Pull the required model before running analysis:
      ```bash
      ollama pull mistral
      ```
1. Place audio files in `data/`.
2. Run transcription for generating transcript:
   ```bash
   python src/transcribe.py
    ```
   --> generates the enriched transcript in ```/transcripts```
3. Run analysis for getting the output:
   ```bash
   python src/analyze.py
   ```
   --> output JSON in ```/outputs/```
4. Run evaluation (if ground truth JSON is available):
   ```bash
   python src/evaluate.py
   ```
5. Bonus Challenge- Agentic Flow demo:
   ```bash
   cd bonus_agentic_flow
   python run_agent_demo.py
   ```
   --> saves log to ```agentic_log.json``` describing state transitions and actions.

> **Note**: Whisper, Hugging Face models, and Ollama weights are **not included** in this submission.
> They will be automatically downloaded the first time you run the code.
> This may take a few minutes depending on your internet speed, but it only happens once.


--- 
## Project Structure
truth-weaver/<br/>
│<br/>
├── src/ # Main solution<br/>
│ ├── preprocess_audio.py   # Noise reduction, audio prep<br/>
│ ├── transcribe.py         # Transcription with Whisper + DSP event injection<br/>
│ ├── analyze.py            # Extract truths, contradictions, deception patterns<br/>
│ ├── evaluate.py           # Compare output with ground truth (if available)<br/>
│<br/>
├── data/ # Input audio files (.mp3, .wav)<br/>
├── transcripts/ # Generated enriched transcript (.txt)<br/>
├── outputs/ # Final structured JSON analyses<br/>
│<br/>
├── bonus_agentic_flow/ # Bonus Challenge solution<br/>
│ ├── agent_flow.py # FSM agent<br/>
│ ├── run_agent_demo.py # Demo linking transcripts → signals → agent actions<br/>
│ ├── agentic_log.json # Full log of an interview run<br/>
│ ├── agent_flow.yaml # Structured state-transition definition<br/>
│ └── flow_diagram.png # State diagram<br/>

---

## Pipeline

### 1. Preprocessing
- Uses `librosa` + `noisereduce` to denoise audio.  
- Normalizes volume, trims silence.

### 2. Transcription
- **Whisper ASR** generates base transcript.  
- Custom **DSP-based event detection** injects markers:  
  - `[whispering]` → low RMS energy  
  - `[sobbing]` → irregular energy & pitch  
  - `[crackle]` → sudden bursts  
  - `[bzzt]` → high-frequency noise  
- Outputs enriched transcripts (closer to hackathon examples).

### 3. Truth Analysis
- LLM-based reasoning + rule-based checks.  
- Identifies:
  - **Revealed truths** (experience, skills, team contribution)
  - **Deception patterns** (inflation, omission, equivocation, contradictions)

### 4. Evaluation
- Compares generated JSON outputs with reference ground truths (if provided).  
- Produces precision/recall/F1 metrics.

---

## Bonus Challenge – Agentic Flow

The **Agentic Flow** models an **adaptive AI interviewer**.  

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

