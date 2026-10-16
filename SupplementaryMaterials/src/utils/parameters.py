# src/utils/parameters.py

class HYPERPARAMS:
    # ================= PATHS =================
    BASE_MODEL_PATH = "./model/Qwen3-VL-4B-Instruct"
    INIT_MODEL_PATH = "./checkpoints/Qwen3-VL-GUI-Init"

    # ================= TRAINING PARAMS FOR SFT =================
    # SFT (Semantic injection)
    SFT_ACTION_SAMPLES = 2000
    SFT_DEFINITION_SAMPLES = 1000
    SFT_SPATIAL_SAMPLES = 2000
    SFT_DATA_PATH = "./data/sft.jsonl"
    QA_DATA_PATH = "./data/qa_squad"
    QA_MIX_RATIO = 0.20
    SFT_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-Loss-Curve"
    
    # Image
    IMAGE_SIZE = 1920
    SFT_MAX_IMAGE_SIDE = 1600
    
    # Optimization
    SFT_LEARN_RATE = 1.5e-5
    SFT_BATCH_SIZE = 1
    SFT_GLOBAL_BATCH_SIZE = 8
    SFT_EPOCHS = 1
    
    # Seeds
    SFT_SEED = 42
    
    # ================= PHASE 2: SFT (ScreenSpot Trajectory) =================
    # Output of Stage 1 is Input of Stage 2
    SFT_2_INPUT_PATH = SFT_OUTPUT_PATH
    SFT_2_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot-New2"
        
    # ScreenSpot Config
    SCREENSPOT_DATA_PATH = "./data/screenspot"
    SCREENSPOT_V2_DATA_PATH = "./data/screenspot_v2"
    SCREENSPOT_TOTAL_SIZE = 1200  # Load 1200 samples total
    SCREENSPOT_TRAIN_RATIO = 0.8  # 80% for Trajectory Training
    # Remaining 20% is reserved as the held-out test split.
        
    SFT_2_LEARN_RATE = 1e-5 # Slightly lower LR for Stage 2
    SFT_2_BATCH_SIZE = 1    # Images are heavy, smaller batch
    SFT_2_EPOCHS = 3        # More epochs to learn visual features

    SFT_3_OUTPUT_PATH = "./checkpoints/Qwen3-VL-GUI-SFT-ScreenSpot-Pro-New2"
    SCREENSPOT_PRO_PATH = "./data/screenspot_pro"
    SCREENSPOT_PRO_TOTAL_SIZE = 1800

    SFT_3_LEARN_RATE = 5e-6 # Slightly lower LR for Stage 2
    SFT_3_BATCH_SIZE = 1    # Images are heavy, smaller batch
    SFT_3_EPOCHS = 2        # More epochs to learn visual features
    
    ACTION_TOKEN_WEIGHT = 20.0

    # ================= EVALUATION PARAMS =================
    VLM_API_KEY = "" # ENTER YOUR API-KEY HERE
    VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    BASELINE_MODEL_NAME = "qwen3-vl-8b-instruct"
    
    # Evaluation Config
    EVAL_DATASET_SIZE = 10 # Number of samples to test
    EVAL_MAX_STEPS = 30
