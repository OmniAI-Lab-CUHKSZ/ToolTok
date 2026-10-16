# src/utils/sft_screenspot.py

from datasets import load_dataset, concatenate_datasets, Image
from .parameters import HYPERPARAMS as HP

class ScreenSpotDataManager:
    """
    Manages loading, shuffling, and strict splitting of ScreenSpot data.
    Ensures no data leakage between Train, Eval, and Test.
    """
    def __init__(self, seed=None):
        self.seed = HP.SFT_SEED if seed is None else seed
        print(f"[ScreenSpotv2] Loading dataset from {HP.SCREENSPOT_V2_DATA_PATH}...")
        try:
            # ScreenSpot usually allows 'test' split download
            # ds = load_dataset(HP.SCREENSPOT_V2_DATA_PATH, split="train") 
            # desktop = load_dataset(
            #     "json",
            #     data_files=f"{HP.SCREENSPOT_V2_DATA_PATH}/screenspot_desktop_v2.json",
            #     split="train"
            # )
            mobile = load_dataset(
                "json",
                data_files=f"{HP.SCREENSPOT_V2_DATA_PATH}/screenspot_mobile_v2.json",
                split="train"
            )
            web = load_dataset(
                "json",
                data_files=f"{HP.SCREENSPOT_V2_DATA_PATH}/screenspot_web_v2.json",
                split="train"
            )
            ds = concatenate_datasets([mobile, web])
            
            IMAGE_DIR = f"{HP.SCREENSPOT_V2_DATA_PATH}/screenspotv2_image"

            def add_image_path(ex):
                ex["image"] = f"{IMAGE_DIR}/{ex['img_filename']}"
                return ex

            ds = ds.map(add_image_path)
            ds = ds.cast_column("image", Image())
            
            print(f"[ScreenSpotv2] Dataset loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load ScreenSpotv2: {e}")
            self.raw_train, self.raw_test = [], []
            return

        # 1. Shuffle (Fixed Seed for reproducibility)
        ds = ds.shuffle(seed=self.seed)
        
        total_available = len(ds)
        limit = min(HP.SCREENSPOT_TOTAL_SIZE, total_available)
        ds = ds.select(range(limit))
        
        print(f"[ScreenSpotv2] Loaded {limit} samples.")

        # 2. Strict Split
        train_count = int(limit * HP.SCREENSPOT_TRAIN_RATIO)
        self.raw_train = ds.select(range(0, train_count))
        self.raw_test = ds.select(range(train_count, limit))
        
        print(f"[ScreenSpotv2] Splits: Train={len(self.raw_train)}, Test={len(self.raw_test)}")
