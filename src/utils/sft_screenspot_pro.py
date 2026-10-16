"""Load the ModelScope ScreenSpot-Pro parquet snapshot from local storage."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from .parameters import HYPERPARAMS as HP


class ScreenSpotDataManager:
    """Deterministically split locally downloaded ScreenSpot-Pro data."""

    def __init__(self, seed=None):
        self.seed = HP.SFT_SEED if seed is None else seed
        root = Path(HP.SCREENSPOT_PRO_PATH)
        print(f"[ScreenSpot Pro] Loading dataset from {root}...")
        parquet_files = sorted(root.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"no ScreenSpot-Pro parquet files found below {root}; download "
                "lmms-lab/ScreenSpot-Pro with ModelScope first"
            )
        dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in parquet_files],
            split="train",
        )
        required = {"image", "instruction", "bbox"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(
                f"unexpected ScreenSpot-Pro schema: missing {sorted(missing)}, "
                f"found {dataset.column_names}"
            )

        dataset = dataset.shuffle(seed=self.seed)
        limit = min(HP.SCREENSPOT_PRO_TOTAL_SIZE, len(dataset))
        dataset = dataset.select(range(limit))
        train_count = int(limit * HP.SCREENSPOT_TRAIN_RATIO)
        self.raw_train = dataset.select(range(train_count))
        self.raw_test = dataset.select(range(train_count, limit))
        print(
            f"[ScreenSpot Pro] Loaded {limit}; splits: "
            f"Train={len(self.raw_train)}, Test={len(self.raw_test)}"
        )
