# Truth-Weaver

## Overview
Truth Weaver is an intelligent system designed to analyze technical interviews.  
It processes **multi-session audio interviews**, extracts **truths vs. deceptions**, and models an **adaptive interviewer (Agentic Flow)**.  

The system has two main parts:
1. **Truth Analysis Pipeline** – Audio → Transcripts → Truth Extraction → Evaluation
2. **Agentic Flow (Bonus Challenge)** – A state-based AI interviewer that adapts to candidate signals.

---

## How to Run

1. Activate the virtual environment
   ```bash
   #For Linux & Mac
   Source venv/bin/activate
   #For Windows
   env/Scripts/activate
   ```
2. Install Dependencies
   ```bash
   pip install -r requirements.txt
    ```
    Additional setup:
    - Ollama must be installed.
    - Pull the required model before running analysis:
      ```bash
      ollama pull mistral
      ```
3. Place audio files in `/Prelims_Source_Code/data/`.
4. For getting Output run the following command
   ```bash
   cd Prelims_Source_Code/
   python main.py
   ```
5. Bonus Challenge- Agentic Flow demo:
   ```bash
   cd Bonus_Challenge
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
├── Prelims_Source_Code/ # Main solution<br/>
│ ├── main.py   # Main file of our project<br/>
│ ├── data/         # You have to add audio dataset here<br/>
│<br/>
├── transcribed.txt # Transcription file having audio converted to text (.txt)<br/>
├── PrelimsSubmission.json # Final structured JSON analyses of all the audio<br/>
├── Readme.md # Readme file<br/>
│<br/>
├── Bonus_Challenge/ # Bonus Challenge solution<br/>
│ ├── agent_flow.py # FSM agent<br/>
│ ├── run_agent_demo.py # Demo linking transcripts → signals → agent actions<br/>
│ ├── agentic_log.json # Full log of an interview run<br/>
│ ├── agent_flow.yaml # Structured state-transition definition<br/>
│ └── flow_diagram.png # State diagram<br/>
| ├── Readme.md # Details of Bonus_Challenge<br/>

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



