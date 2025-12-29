from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sys

# Ensure we can import from pyscripts even when running from repo root
sys.path.append(str(Path(__file__).resolve().parent / "pyscripts"))
from generate_scores import load_data
from smile.smile import SMILE


def print_step(step_no: int, title: str, emoji: str) -> None:
    line = "=" * 64
    print(f"\n{line}\nStep {step_no}: {title} {emoji}\n{line}")


def main() -> None:
    # Sample input file
    input_path = "sample_data/sample_input.json"

    args = SimpleNamespace(
        input_file=str(input_path),
        pred_file=None, # to be used if predictions are present in a separate file
        use_ans=False,
    )

    print_step(1, "Loading sample input", "📥")
    proc_data = load_data(args)
    if isinstance(proc_data, np.ndarray):
        print(f"Loaded rows: {proc_data.shape[0]} 📦")
    else:
        print(f"Loaded rows: {len(proc_data)} 📦")

    print_step(2, "Initializing SMILE", "🙂")
    smile = SMILE(
        emb_model="ember-v1",
        eval_metrics=["avg", "hm"],
        assign_bins=False,
        use_exact_matching=True,
        verbose=True,
    )

    print_step(3, "Computing SMILE scores", "🧮⚙️")
    results = smile.generate_scores(proc_data)

    # Summaries
    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if isinstance(arr, np.ndarray) else float(np.mean(np.array(arr)))

    sent_mean = _mean(results["sent_emb_scores"]) if "sent_emb_scores" in results else float("nan")
    kwd_mean = _mean(results["kwd_scores"]) if "kwd_scores" in results else float("nan")
    avg_mean = _mean(results["avg"]) if "avg" in results else float("nan")
    hm_mean = _mean(results["hm"]) if "hm" in results else float("nan")

    print_step(4, "Results summary", "📊")
    print(f"Sentence embedding score (mean): {sent_mean:.4f} ✨")
    print(f"Keyword score (mean): {kwd_mean:.4f} 🔑")
    print(f"SMILE avg (mean): {avg_mean:.4f} 😊")
    print(f"SMILE hm  (mean): {hm_mean:.4f} 🤝")

    if hasattr(proc_data, "shape") and proc_data.shape[0] > 0:
        print("\n-- First item details 🔎 --")
        print(f"question: {proc_data[0, 0]}")
        print(f"answer  : {proc_data[0, 1]}")
        print(f"syn_ans : {proc_data[0, 2]}")
        print(f"pred    : {proc_data[0, 3]}")
        print(f"sent_emb_score: {results['sent_emb_scores'][0]:.4f} ✨")
        print(f"kwd_score     : {results['kwd_scores'][0]:.4f} 🔑")
        print(f"avg           : {results['avg'][0]:.4f} 😊")
        print(f"hm            : {results['hm'][0]:.4f} 🤝")


if __name__ == "__main__":
    main()
