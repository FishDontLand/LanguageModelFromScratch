import argparse
import json
from importlib.resources import files
from pathlib import Path

SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def prepare_hendrycks_math_data(data_dir: str | Path = "data") -> None:
    from datasets import load_dataset

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    prompt = files("cs336.post_training").joinpath("prompts", "r1_zero.prompt").read_text(encoding="utf-8")
    datasets = [load_dataset("EleutherAI/hendrycks_math", subject) for subject in SUBJECTS]

    for split in ("train", "test"):
        with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as output:
            for dataset in datasets:
                for example in dataset[split]:
                    row = {
                        "question": prompt.replace("{question}", example["problem"]),
                        "answer": example["solution"],
                    }
                    output.write(json.dumps(row) + "\n")


def prepare_model(cache_path: str | Path, model_id: str = "Qwen/Qwen2.5-Math-1.5B") -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, cache_dir=cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MATH data for post-training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    prepare_hendrycks_math_data(args.data_dir)


if __name__ == "__main__":
    main()
