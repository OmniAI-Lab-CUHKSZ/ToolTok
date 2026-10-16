"""Configurable evaluation entry point for ToolTok GUI agents.

Examples:
    python -m src.eval.evaluate --help
    python -m src.eval.evaluate --model-path ./checkpoints/my-model --limit 100
    python -m src.eval.evaluate --mode api_baseline --temperature 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ..tools.runner import Runner
from ..tools.visual_utils import visualize_trajectory
from ..utils.parameters import HYPERPARAMS as HP

try:
    from .baseline_vlm import BaselineAPIRunner
except ImportError:
    BaselineAPIRunner = None

try:
    from .baseline_grounding import BaselineGroundingRunner
except ImportError:
    BaselineGroundingRunner = None


EVAL_MODES = ("trained", "api_baseline", "grounding_baseline")
DATASETS = ("screenspot", "screenspot_pro", "screenspot_v2")
BBOX_FORMATS = ("auto", "xyxy", "xywh")
DTYPES = ("auto", "bfloat16", "float16", "float32")


@dataclass(frozen=True)
class EvaluationConfig:
    """All inputs that can affect one evaluation run."""

    mode: str = "trained"
    model_path: str = HP.SFT_3_OUTPUT_PATH
    api_model: str = HP.BASELINE_MODEL_NAME
    dataset: str = "screenspot"
    limit: Optional[int] = getattr(HP, "EVAL_DATASET_SIZE", 10)
    bbox_expansion: float = getattr(HP, "EVAL_BBOX_EXPANSION", 0.0)
    bbox_format: str = "xyxy"
    max_steps: int = HP.EVAL_MAX_STEPS
    seed: int = HP.SFT_SEED
    temperature: float = 0.0
    output_dir: str = "./results/eval"
    run_name: Optional[str] = None
    save_images: bool = True
    dtype: str = "bfloat16"
    device_map: Optional[str] = "auto"

    def __post_init__(self) -> None:
        if self.mode not in EVAL_MODES:
            raise ValueError(f"mode must be one of {EVAL_MODES}, got {self.mode!r}")
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.bbox_format not in BBOX_FORMATS:
            raise ValueError(
                f"bbox_format must be one of {BBOX_FORMATS}, got {self.bbox_format!r}"
            )
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than zero, or omitted for all samples")
        if not 0.0 <= self.bbox_expansion <= 1.0:
            raise ValueError("bbox_expansion must be between 0.0 and 1.0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")


def set_seed(seed: int) -> None:
    """Seed all random number generators used by the evaluation path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_dataset(dataset_name: str, seed: int):
    """Load one supported dataset and return its held-out test split."""
    if dataset_name == "screenspot":
        from ..utils.sft_screenspot import ScreenSpotDataManager

        manager = ScreenSpotDataManager(seed=seed)
    elif dataset_name == "screenspot_pro":
        from ..utils.sft_screenspot_pro import ScreenSpotDataManager

        manager = ScreenSpotDataManager(seed=seed)
    elif dataset_name == "screenspot_v2":
        from ..utils.sft_screenspot_v2 import ScreenSpotDataManager

        manager = ScreenSpotDataManager(seed=seed)
    else:  # Guarded by EvaluationConfig; retained for direct function callers.
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = manager.raw_test
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset {dataset_name!r} has an empty test split")
    return dataset


def select_samples(dataset, limit: Optional[int]):
    """Select at most ``limit`` rows from either a HF Dataset or a sequence."""
    if limit is None or limit >= len(dataset):
        return dataset

    indices = range(limit)
    if hasattr(dataset, "select"):
        return dataset.select(indices)
    return [dataset[index] for index in indices]


def load_sample_image(sample: dict[str, Any]) -> Image.Image:
    """Load a PIL image from the image_path or image field of a sample."""
    image_path = sample.get("image_path")
    if image_path:
        try:
            return Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load image_path {image_path!r}: {error}") from error

    image_data = sample.get("image")
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data).convert("RGB")
    if isinstance(image_data, dict) and image_data.get("bytes") is not None:
        from io import BytesIO

        return Image.open(BytesIO(image_data["bytes"])).convert("RGB")
    if image_data is not None:
        try:
            return Image.open(image_data).convert("RGB")
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"cannot load sample image: {error}") from error

    raise ValueError("sample contains neither a usable image_path nor image")


def prepare_bbox(
    sample: dict[str, Any],
    image_size: tuple[int, int],
    bbox_format: str,
    expansion: float,
) -> list[float]:
    """Convert a sample bbox to clamped absolute ``[x1, y1, x2, y2]`` pixels.

    All bundled ScreenSpot datasets use xyxy. ``auto`` therefore resolves to
    xyxy as well; ``xywh`` remains available for explicitly supplied datasets.
    """
    width, height = image_size
    raw_bbox = sample.get("bbox")
    bbox_missing = raw_bbox is None or len(raw_bbox) == 0
    is_point = bbox_missing and sample.get("point") is not None
    if is_point:
        point = sample["point"]
        raw_bbox = [point[0], point[1], point[0], point[1]]

    if raw_bbox is None or len(raw_bbox) != 4:
        raise ValueError(f"expected a four-value bbox or two-value point, got {raw_bbox!r}")

    bbox = [float(value) for value in raw_bbox]
    normalized = all(0.0 <= value <= 1.0 for value in bbox)
    resolved_format = bbox_format
    if is_point:
        resolved_format = "xyxy"
    elif bbox_format == "auto":
        resolved_format = "xyxy"

    x1, y1, third, fourth = bbox
    if resolved_format == "xywh":
        x2, y2 = x1 + third, y1 + fourth
    else:
        x2, y2 = third, fourth

    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    margin_x = width * expansion
    margin_y = height * expansion
    return [
        max(0.0, x1 - margin_x),
        max(0.0, y1 - margin_y),
        min(float(width), x2 + margin_x),
        min(float(height), y2 + margin_y),
    ]


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def initialize_runner(config: EvaluationConfig):
    """Create the selected runner and, for local evaluation, its model stack."""
    if config.mode == "api_baseline":
        if BaselineAPIRunner is None:
            raise ImportError("BaselineAPIRunner dependencies are unavailable")
        return (
            BaselineAPIRunner(
                temperature=config.temperature,
                model_name=config.api_model,
            ),
            None,
            None,
        )

    if config.mode == "grounding_baseline":
        if BaselineGroundingRunner is None:
            raise ImportError("BaselineGroundingRunner dependencies are unavailable")
        return (
            BaselineGroundingRunner(
                temperature=config.temperature,
                model_name=config.api_model,
            ),
            None,
            None,
        )

    model_path = Path(config.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    print(f"[EVAL] Loading local model from: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        device_map=config.device_map,
        torch_dtype=resolve_torch_dtype(config.dtype),
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    return Runner(), model, processor


def create_result_dir(config: EvaluationConfig, timestamp: str) -> tuple[Path, Path]:
    if config.run_name:
        run_name = config.run_name
    else:
        relaxation = round(config.bbox_expansion * 100)
        run_name = (
            f"{config.dataset}_{config.mode}_{timestamp}"
            f"_relax{relaxation}_seed{config.seed}"
        )

    result_dir = Path(config.output_dir).expanduser() / run_name
    image_dir = result_dir / "images"
    result_dir.mkdir(parents=True, exist_ok=True)
    if config.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, image_dir


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run an evaluation and return the same report that is written to disk."""
    set_seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir, image_dir = create_result_dir(config, timestamp)

    print("\n[EVAL] Configuration")
    print(json.dumps(asdict(config), indent=2, ensure_ascii=False))
    print(f"[EVAL] Results: {result_dir}")

    dataset = load_test_dataset(config.dataset, seed=config.seed)
    dataset = select_samples(dataset, config.limit)
    print(f"[EVAL] Selected samples: {len(dataset)}")

    runner, model, processor = initialize_runner(config)
    details: list[dict[str, Any]] = []
    counters = {
        "success_count": 0,
        "total_steps": 0,
        "error_count": 0,
        "failed_reasons": {1: 0, 2: 0, 3: 0, 4: 0},
    }

    for index, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
        try:
            image = load_sample_image(sample)
            final_bbox = prepare_bbox(
                sample=sample,
                image_size=image.size,
                bbox_format=config.bbox_format,
                expansion=config.bbox_expansion,
            )
            instruction = sample["instruction"]
            ground_truth_data = [
                {
                    "image": image,
                    "instruction": instruction,
                    "action_type": "click",
                    "bbox": final_bbox,
                }
            ]

            if config.mode == "trained":
                trajectory, _, _ = runner.run_trajectory(
                    model=model,
                    processor=processor,
                    input_text=instruction,
                    ground_truth_data=ground_truth_data,
                    max_steps=config.max_steps,
                    temperature=config.temperature,
                )
            else:
                trajectory, _, _ = runner.run_trajectory(
                    input_text=instruction,
                    ground_truth_data=ground_truth_data,
                    max_steps=config.max_steps,
                )

            success = trajectory.failed == 0
            counters["total_steps"] += trajectory.step_count
            if success:
                counters["success_count"] += 1
            elif trajectory.failed in counters["failed_reasons"]:
                counters["failed_reasons"][trajectory.failed] += 1

            image_filename = None
            if config.save_images:
                status = "PASS" if success else "FAIL"
                image_filename = f"{config.mode}_id_{index:04d}_{status}.png"
                visualization = visualize_trajectory(
                    base_image=image,
                    cursor_path=trajectory.cursor_path,
                    actions=trajectory.tools,
                    gt_bbox=final_bbox,
                    success=success,
                    instruction=trajectory.global_question,
                )
                visualization.save(image_dir / image_filename)

            details.append(
                {
                    "id": index,
                    "instruction": instruction,
                    "success": success,
                    "steps": trajectory.step_count,
                    "fail_code": trajectory.failed,
                    "history": trajectory.tools,
                    "gt_bbox_xyxy": final_bbox,
                    "vis_file": image_filename,
                }
            )

            if index % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            counters["error_count"] += 1
            print(f"[EVAL] Sample {index} failed: {error}")

    processed_count = len(details)
    accuracy = counters["success_count"] / processed_count if processed_count else 0.0
    average_steps = counters["total_steps"] / processed_count if processed_count else 0.0

    report = {
        "meta": {
            **asdict(config),
            "timestamp": timestamp,
            "selected_samples": len(dataset),
            "processed_samples": processed_count,
        },
        "metrics": {
            "accuracy": accuracy,
            "avg_steps": average_steps,
            "success_count": counters["success_count"],
            "error_count": counters["error_count"],
            "failed_breakdown": counters["failed_reasons"],
        },
        "details": details,
    }

    report_path = result_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 48)
    print(f"EVAL REPORT: {config.mode.upper()} / {config.dataset}")
    print(f"Selected:      {len(dataset)}")
    print(f"Processed:     {processed_count}")
    print(f"Errors:        {counters['error_count']}")
    print(f"Success Rate:  {accuracy:.2%}")
    print(f"Avg Steps:     {average_steps:.2f}")
    print(f"Failure Stats: {counters['failed_reasons']}")
    print(f"Report:        {report_path}")
    print("=" * 48 + "\n")
    return report


def evaluate_model(
    mode: str = "trained",
    limit: Optional[int] = None,
    model_path: Optional[str] = None,
    bbox_expansion: Optional[float] = None,
    *,
    api_model: str = HP.BASELINE_MODEL_NAME,
    dataset: str = "screenspot",
    bbox_format: str = "xyxy",
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    temperature: float = 0.0,
    output_dir: str = "./results/eval",
    run_name: Optional[str] = None,
    save_images: bool = True,
    dtype: str = "bfloat16",
    device_map: Optional[str] = "auto",
) -> dict[str, Any]:
    """Backward-compatible Python API around :func:`run_evaluation`."""
    config = EvaluationConfig(
        mode=mode,
        model_path=model_path or HP.SFT_3_OUTPUT_PATH,
        api_model=api_model,
        dataset=dataset,
        limit=limit,
        bbox_expansion=(
            getattr(HP, "EVAL_BBOX_EXPANSION", 0.0)
            if bbox_expansion is None
            else bbox_expansion
        ),
        bbox_format=bbox_format,
        max_steps=HP.EVAL_MAX_STEPS if max_steps is None else max_steps,
        seed=HP.SFT_SEED if seed is None else seed,
        temperature=temperature,
        output_dir=output_dir,
        run_name=run_name,
        save_images=save_images,
        dtype=dtype,
        device_map=device_map,
    )
    return run_evaluation(config)


def optional_positive_int(value: str) -> Optional[int]:
    if value.lower() in {"all", "none"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolTok or a baseline GUI agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=EVAL_MODES, default="trained")
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=HP.SFT_3_OUTPUT_PATH,
    )
    parser.add_argument(
        "--api-model",
        default=HP.BASELINE_MODEL_NAME,
        help="remote model name used by API baseline modes",
    )
    parser.add_argument("--dataset", choices=DATASETS, default="screenspot")
    parser.add_argument(
        "--limit",
        "--max-samples",
        dest="limit",
        type=optional_positive_int,
        default=getattr(HP, "EVAL_DATASET_SIZE", 10),
        help="maximum test samples; use 'all' for the entire test split",
    )
    parser.add_argument(
        "--bbox-expansion",
        "--relaxation",
        dest="bbox_expansion",
        type=float,
        default=getattr(HP, "EVAL_BBOX_EXPANSION", 0.0),
        help="fraction of image width/height added around each target bbox",
    )
    parser.add_argument(
        "--bbox-format",
        choices=BBOX_FORMATS,
        default="xyxy",
        help="source bbox convention before conversion to absolute xyxy",
    )
    parser.add_argument("--max-steps", type=int, default=HP.EVAL_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=HP.SFT_SEED)
    parser.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="./results/eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map; use 'none' to disable it",
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> EvaluationConfig:
    args = build_parser().parse_args(argv)
    device_map = None if args.device_map.lower() == "none" else args.device_map
    return EvaluationConfig(
        mode=args.mode,
        model_path=args.model_path,
        api_model=args.api_model,
        dataset=args.dataset,
        limit=args.limit,
        bbox_expansion=args.bbox_expansion,
        bbox_format=args.bbox_format,
        max_steps=args.max_steps,
        seed=args.seed,
        temperature=args.temperature,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_images=not args.no_save_images,
        dtype=args.dtype,
        device_map=device_map,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_evaluation(parse_config(argv))


if __name__ == "__main__":
    main()
