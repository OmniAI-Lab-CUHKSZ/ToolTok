# src/utils/sft_screenspot.py

from pathlib import Path

from datasets import load_dataset
from .parameters import HYPERPARAMS as HP

class ScreenSpotDataManager:
    """
    Manages loading, shuffling, and strict splitting of ScreenSpot data.
    Ensures no data leakage between Train, Eval, and Test.
    """
    def __init__(self, seed=None):
        self.seed = HP.SFT_SEED if seed is None else seed
        print(f"[ScreenSpot] Loading dataset from {HP.SCREENSPOT_DATA_PATH}...")
        root = Path(HP.SCREENSPOT_DATA_PATH)
        parquet_files = sorted(root.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"no ScreenSpot parquet files found below {root}; download "
                "rootsautomation/ScreenSpot with ModelScope first"
            )
        ds = load_dataset(
            "parquet",
            data_files=[str(path) for path in parquet_files],
            split="train",
        )
        required = {"image", "instruction"}
        missing = required.difference(ds.column_names)
        if missing:
            raise ValueError(
                f"unexpected ScreenSpot schema: missing {sorted(missing)}, "
                f"found {ds.column_names}"
            )

        # 1. Shuffle (Fixed Seed for reproducibility)
        ds = ds.shuffle(seed=self.seed)
        
        total_available = len(ds)
        limit = min(HP.SCREENSPOT_TOTAL_SIZE, total_available)
        ds = ds.select(range(limit))
        
        print(f"[ScreenSpot] Loaded {limit} samples.")

        # 2. Strict Split
        train_count = int(limit * HP.SCREENSPOT_TRAIN_RATIO)
        self.raw_train = ds.select(range(0, train_count))
        self.raw_test = ds.select(range(train_count, limit))
        
        print(f"[ScreenSpot] Splits: Train={len(self.raw_train)}, Test={len(self.raw_test)}")
