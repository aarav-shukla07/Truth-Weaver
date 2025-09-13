import os
import json
import Levenshtein

# Paths
TRANSCRIPTS_DIR = "transcripts"
GROUND_TRUTH_DIR = "ground_truth"   # expected ground truth files
OUTPUTS_DIR = "outputs"

def transcript_score(pred, truth):
    return 1 - (Levenshtein.distance(pred, truth) / max(len(pred), len(truth)))

def evaluate_transcripts():
    if not os.path.exists(GROUND_TRUTH_DIR):
        print("[!] No ground_truth/ folder found. Skipping transcript evaluation.")
        return

    scores = []
    for file in os.listdir(TRANSCRIPTS_DIR):
        if file.endswith(".txt"):
            pred_file = os.path.join(TRANSCRIPTS_DIR, file)
            truth_file = os.path.join(GROUND_TRUTH_DIR, file)

            if os.path.exists(truth_file):
                with open(pred_file, "r", encoding="utf-8") as f1, \
                     open(truth_file, "r", encoding="utf-8") as f2:
                    score = transcript_score(f1.read(), f2.read())
                    scores.append(score)
                    print(f"{file}: {score:.3f}")
            else:
                print(f"[!] Ground truth missing for {file}")

    if scores:
        print(f"\nAverage Transcript Score: {sum(scores)/len(scores):.3f}")
    else:
        print("[!] No transcript scores calculated.")

def evaluate_json(pred_file, truth_file):
    if not os.path.exists(truth_file):
        print(f"[!] Ground truth JSON {truth_file} not found. Skipping JSON evaluation.")
        return

    with open(pred_file, "r", encoding="utf-8") as f1, \
         open(truth_file, "r", encoding="utf-8") as f2:
        pred = json.load(f1)
        truth = json.load(f2)

    pred_keys = set(json.dumps(pred).split())
    truth_keys = set(json.dumps(truth).split())
    jaccard = len(pred_keys & truth_keys) / len(pred_keys | truth_keys)
    print(f"\nJaccard Similarity (JSON): {jaccard:.3f}")

if __name__ == "__main__":
    print("=== Transcript Evaluation ===")
    evaluate_transcripts()

    print("\n=== JSON Evaluation ===")
    evaluate_json("outputs/phoenix_2024.json", "ground_truth/phoenix_2024.json")
