"""Build the TBLite Beaker datasets from the pinned Hugging Face revision.
Run: uv run --no-project --with pyarrow python .beaker/build_tblite_dataset.py [--split-only|--local-smoke|--upload]"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

HF_REPO = "NousResearch/openthoughts-tblite"
HF_REVISION = "44c975f590dde88316572d7e2a779ec1112d4a4b"
PARQUET_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/data/train-00000-of-00001.parquet"
SPLITS_FILE = Path(__file__).with_name("tblite_splits.json")
SEED = 20261102
TEST_FRACTION = 0.3
QUICKSTART_TRAIN, QUICKSTART_TEST = 6, 3
# The source mixes "hard"/"Hard" and "expert"/"very-hard".
DIFFICULTY_ALIASES = {"very-hard": "expert"}
AGENT_KEY = "hermes-tblite"


def normalized_difficulty(raw: str) -> str:
    label = raw.strip().lower()
    return DIFFICULTY_ALIASES.get(label, label)


def load_records(parquet_path: Path) -> dict[str, dict]:
    import pyarrow.parquet as pq

    return {record["task_name"]: record for record in pq.read_table(parquet_path).to_pylist()}


def compute_splits(records: dict[str, dict]) -> dict:
    rng = random.Random(SEED)
    by_difficulty: dict[str, list[str]] = defaultdict(list)
    for name, record in sorted(records.items()):
        by_difficulty[normalized_difficulty(record["difficulty"])].append(name)
    train, test = [], []
    for names in by_difficulty.values():
        rng.shuffle(names)
        test_count = round(len(names) * TEST_FRACTION)
        test += names[:test_count]
        train += names[test_count:]

    def quick(pool: list[str], count: int) -> list[str]:
        fast = sorted(n for n in pool if normalized_difficulty(records[n]["difficulty"]) in {"easy", "medium"}
                      and records[n]["agent_timeout_sec"] <= 900)
        return sorted(rng.sample(fast, count))

    return {
        "source": {"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "seed": SEED},
        "full": {"train": sorted(train), "test": sorted(test)},
        "quickstart": {"train": quick(train, QUICKSTART_TRAIN), "test": quick(test, QUICKSTART_TEST)},
    }


def to_row(record: dict) -> dict:
    return {
        "id": record["task_name"],
        "input": {
            "task_name": record["task_name"],
            "instruction": record["instruction"],
            "docker_image": record["docker_image"],
            "category": record["category"],
            "difficulty": normalized_difficulty(record["difficulty"]),
            "agent_timeout_sec": int(record["agent_timeout_sec"]),
            "test_timeout_sec": int(record["test_timeout_sec"]),
        },
        "expected": {"test_sh": record["test_sh"]},
        "tests_tar_b64": record["tests_tar"],
        "metadata": {"category": record["category"], "difficulty": normalized_difficulty(record["difficulty"])},
    }


def write_dataset(directory: Path, records: dict[str, dict], split: dict[str, list[str]], name: str) -> None:
    for split_name, names in split.items():
        with (directory / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
            for task_name in names:
                output.write(json.dumps(to_row(records[task_name])) + "\n")
    manifest = {"name": name, "hf_repo": HF_REPO, "hf_revision": HF_REVISION, "split_seed": SEED,
                "splits": {k: len(v) for k, v in split.items()}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def upload(directory: Path, name: str, split: dict[str, list[str]]) -> str:
    command = ["beaker", "dataset", "upload", str(directory), "--name", name,
               "--total-count", str(sum(len(v) for v in split.values())), "--agent", AGENT_KEY, "--json"]
    for split_name, names in split.items():
        command += ["--split", f"{split_name}={len(names)}"]
    artifact = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return f"{artifact['artifact_key']}@{artifact['dataset_revision']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--local-smoke", action="store_true", help="run beaker smoke against the quickstart split")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="beaker-tblite-") as temp:
        parquet = Path(temp) / "tblite.parquet"
        urllib.request.urlretrieve(PARQUET_URL, parquet)
        records = load_records(parquet)
        if args.split_only:
            SPLITS_FILE.write_text(json.dumps(compute_splits(records), indent=2) + "\n", encoding="utf-8")
            return
        splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
        for variant in ("quickstart", "full"):
            directory = Path(temp) / variant
            directory.mkdir()
            write_dataset(directory, records, splits[variant], f"tblite-{variant}")
        if args.local_smoke:
            subprocess.run(["beaker", "run", "smoke", "--strict", "--integration-id", "tblite", "--config",
                            json.dumps({"local_dataset_path": str(Path(temp) / "quickstart")})], check=True)
        if args.upload:
            for variant in ("quickstart", "full"):
                print(variant, upload(Path(temp) / variant, f"tblite-{variant}", splits[variant]))


if __name__ == "__main__":
    main()
