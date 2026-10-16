"""Local SQuAD supervision and deterministic dataset mixing helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from .parameters import HYPERPARAMS as HP


QA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question accurately and concisely "
    "using the supplied context."
)


def _first_answer_text(answers: Any) -> str:
    """Extract the first non-empty answer from a SQuAD answer object."""
    if isinstance(answers, dict):
        answers = answers.get("text", [])
    if isinstance(answers, str):
        answers = [answers]
    if isinstance(answers, (list, tuple)):
        for answer in answers:
            if str(answer).strip():
                return str(answer).strip()
    raise ValueError(f"SQuAD row has no usable answer: {answers!r}")


def load_local_squad(data_path: str | Path, seed: int | None = None):
    """Load and deterministically shuffle ModelScope-downloaded SQuAD parquet."""
    root = Path(data_path)
    parquet_files = sorted(root.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"no SQuAD parquet files found below {root}; run the ModelScope "
            "dataset downloader first"
        )
    train_files = [path for path in parquet_files if "train" in path.name.lower()]
    files = train_files or parquet_files
    dataset = load_dataset(
        "parquet", data_files=[str(path) for path in files], split="train"
    )
    required = {"context", "question", "answers"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"unexpected SQuAD schema in {root}: missing {sorted(missing)}, "
            f"found {dataset.column_names}"
        )
    return dataset.shuffle(seed=HP.SFT_SEED if seed is None else seed)


class QASFTDataset(Dataset):
    """Ordinary language-model supervision with no GUI action target."""

    def __init__(self, data_path: str | Path, num_samples: int, seed: int | None = None):
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.data = load_local_squad(data_path, seed=seed)
        if not self.data:
            raise ValueError(f"SQuAD dataset at {data_path} is empty")
        self.num_samples = num_samples
        # Keep a single neutral image template so multimodal batch shapes stay uniform.
        self.neutral_image = Image.new("RGB", (64, 64), color=(0, 0, 0))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data[index % len(self.data)]
        user_content = (
            "Context:\n"
            f"{str(row['context']).strip()}\n\n"
            "Question:\n"
            f"{str(row['question']).strip()}"
        )
        return {
            "messages": [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": _first_answer_text(row["answers"])},
            ],
            "image": self.neutral_image.copy(),
            "requires_action": False,
        }


class DeterministicMixedDataset(Dataset):
    """Interleave primary and QA rows with a reproducible exact sample count."""

    def __init__(self, primary: Dataset, qa: Dataset, qa_fraction: float):
        if not 0.0 < qa_fraction < 1.0:
            raise ValueError(f"qa_fraction must be between 0 and 1, got {qa_fraction}")
        if len(primary) < 1:
            raise ValueError("primary dataset is empty")
        expected_qa = math.ceil(len(primary) * qa_fraction / (1.0 - qa_fraction))
        if len(qa) != expected_qa:
            raise ValueError(
                f"QA dataset has {len(qa)} rows; expected {expected_qa} for "
                f"primary={len(primary)}, fraction={qa_fraction}"
            )
        self.primary = primary
        self.qa = qa
        self.qa_fraction = qa_fraction
        # Evenly distribute QA positions without relying on process-local RNG state.
        total = len(primary) + len(qa)
        qa_positions = {
            min(total - 1, math.floor((index + 0.5) * total / len(qa)))
            for index in range(len(qa))
        }
        if len(qa_positions) != len(qa):
            raise RuntimeError("failed to construct unique QA interleave positions")
        self._index = []
        primary_index = qa_index = 0
        for position in range(total):
            if position in qa_positions:
                self._index.append((True, qa_index))
                qa_index += 1
            else:
                self._index.append((False, primary_index))
                primary_index += 1

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int):
        is_qa, source_index = self._index[index]
        return self.qa[source_index] if is_qa else self.primary[source_index]


def mix_with_squad(primary: Dataset, data_path: str | Path, qa_fraction: float, seed: int):
    """Build a primary/QA mixture whose QA share is ``qa_fraction``."""
    qa_count = math.ceil(len(primary) * qa_fraction / (1.0 - qa_fraction))
    qa = QASFTDataset(data_path, num_samples=qa_count, seed=seed)
    mixed = DeterministicMixedDataset(primary, qa, qa_fraction=qa_fraction)
    actual_fraction = len(qa) / len(mixed)
    print(
        f"[QA Mix] Primary={len(primary)}, SQuAD={len(qa)}, "
        f"combined={len(mixed)}, QA={actual_fraction:.2%}"
    )
    return mixed
