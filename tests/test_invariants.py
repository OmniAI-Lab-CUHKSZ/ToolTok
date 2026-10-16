import json
import importlib
import os
import pkgutil
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from src.train.action_trainer import WeightedActionTrainer, resolve_gradient_accumulation
from src.utils.action import ACTION_TOKENS, atomic_action_token_ids
from src.utils.action_logic import MOVE_SPECS, get_move_delta
from src.utils.action_format import extract_action_response, format_action_response
from src.utils.pathing import build_action_path, point_in_bbox, sample_start_position
from src.utils.qa_data import DeterministicMixedDataset
from src.utils.spatial_cot import (
    TASK_OPENERS,
    format_spatial_cot,
    format_task_opener,
)


class _RangeDataset(Dataset):
    def __init__(self, size, source):
        self.size = size
        self.source = source

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.source, index


class ActionFormatTests(unittest.TestCase):
    def test_concise_spatial_cot_is_reproducible_and_label_faithful(self):
        arguments = {
            "task": "Click the Settings button",
            "cursor_pos": (100, 100),
            "target_bbox": (500, 300, 600, 400),
            "image_size": (1000, 500),
            "action_token": "<MOVE_DOWN_RIGHT_FAR>",
        }
        first = format_spatial_cot(**arguments)
        self.assertEqual(first, format_spatial_cot(**arguments))
        reasoning, action = extract_action_response(first)
        self.assertEqual(action, "<MOVE_DOWN_RIGHT_FAR>")
        self.assertIn("Click the Settings button", reasoning)
        self.assertIn("40%", reasoning)
        self.assertIn("25%", reasoning)
        self.assertNotIn("grid", reasoning.lower())
        self.assertNotIn("farthest", reasoning.lower())

    def test_task_opener_has_many_surface_forms(self):
        self.assertGreaterEqual(len(TASK_OPENERS), 12)
        variants = {
            format_task_opener("Click Settings", random.Random(seed))
            for seed in range(200)
        }
        self.assertEqual(len(variants), len(TASK_OPENERS))
        for variant in variants:
            self.assertIn("Click Settings", variant)
            self.assertTrue(
                "locate" in variant.lower()
                or "find" in variant.lower()
                or "identify" in variant.lower()
            )

    def test_click_cot_requires_bbox_overlap(self):
        with self.assertRaises(ValueError):
            format_spatial_cot(
                "Click Settings",
                cursor_pos=(0, 0),
                target_bbox=(100, 100, 200, 200),
                image_size=(500, 500),
                action_token="<CLICK_SHORT>",
            )

    def test_stage_one_covers_the_complete_action_vocabulary(self):
        from src.utils.generate_data import TOKEN_META

        self.assertEqual(set(TOKEN_META), set(ACTION_TOKENS))

    def test_round_trip_requires_one_final_action(self):
        response = format_action_response("Move toward the target.", "<MOVE_RIGHT_CLO>")
        self.assertEqual(
            extract_action_response(response),
            ("Move toward the target.", "<MOVE_RIGHT_CLO>"),
        )
        self.assertEqual(extract_action_response(response + " trailing"), ("Move toward the target.", None))

    def test_generated_stage_one_data(self):
        data_path = Path("data/sft.jsonl")
        if not data_path.exists():
            self.skipTest("generate data/sft.jsonl first")
        with data_path.open(encoding="utf-8") as data_file:
            records = [json.loads(line) for line in data_file]
        self.assertEqual(len(records), 3000)
        covered_actions = set()
        for record in records:
            response = record["messages"][-1]["content"]
            reasoning, action = extract_action_response(response)
            self.assertTrue(reasoning)
            self.assertIn(action, ACTION_TOKENS)
            self.assertTrue(response.endswith(action))
            covered_actions.add(action)
            if record["data_type"] == "action":
                instruction = record["messages"][0]["content"].removeprefix(
                    "[Action] "
                )
                self.assertIn(instruction, reasoning)
        self.assertEqual(covered_actions, set(ACTION_TOKENS))

    def test_diagonal_action_vocabulary_is_complete(self):
        expected = {
            f"<MOVE_{direction}_{distance}>"
            for direction in ("UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")
            for distance in ("FAR", "MID", "CLO")
        }
        self.assertEqual(expected, expected.intersection(ACTION_TOKENS))
        self.assertEqual(len(MOVE_SPECS), 8 * 3)
        self.assertEqual(
            set(MOVE_SPECS),
            {token for token in ACTION_TOKENS if token.startswith("<MOVE_")},
        )


class PathingTests(unittest.TestCase):
    def test_move_deltas_scale_with_each_screen_axis(self):
        image_size = (1000, 500)
        self.assertEqual(get_move_delta("<MOVE_RIGHT_FAR>", image_size), (250, 0))
        self.assertEqual(get_move_delta("<MOVE_UP_MID>", image_size), (0, -50))
        self.assertEqual(
            get_move_delta("<MOVE_DOWN_LEFT_CLO>", image_size), (-20, 10)
        )
        self.assertEqual(
            get_move_delta("<MOVE_UP_RIGHT_FAR>", image_size), (250, -125)
        )

    def test_diagonal_path_uses_a_diagonal_action(self):
        path = build_action_path(
            (0, 0),
            (790, 390, 810, 410),
            (1000, 500),
        )
        self.assertTrue(path)
        self.assertTrue(path[0][0].startswith("<MOVE_DOWN_RIGHT_"), path[0])

    def test_path_uses_screen_normalized_geometry(self):
        path = build_action_path(
            (100, 100),
            (1510, 800, 1550, 840),
            (1920, 1080),
        )
        self.assertTrue(path)
        self.assertTrue(path[0][0].startswith("<MOVE_DOWN_RIGHT_"), path[0])

    def test_click_labels_are_always_inside_bbox(self):
        rng = random.Random(42)
        click_count = 0
        for _ in range(2000):
            width, height = rng.randint(500, 2000), rng.randint(500, 2000)
            x1, y1 = rng.randint(0, width - 20), rng.randint(0, height - 20)
            bbox = (
                x1,
                y1,
                min(width - 1, x1 + rng.randint(5, 200)),
                min(height - 1, y1 + rng.randint(5, 200)),
            )
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            start = sample_start_position(center, (width, height), rng)
            for token, position in build_action_path(start, bbox, (width, height)):
                if token == "<CLICK_SHORT>":
                    click_count += 1
                    self.assertTrue(point_in_bbox(position, bbox))
        self.assertGreater(click_count, 0)


class RuntimeTests(unittest.TestCase):
    def test_qa_mix_is_exactly_twenty_percent(self):
        mixed = DeterministicMixedDataset(
            _RangeDataset(8, "gui"),
            _RangeDataset(2, "qa"),
            qa_fraction=0.2,
        )
        rows = [mixed[index] for index in range(len(mixed))]
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(source == "qa" for source, _ in rows), 2)

    def test_trainer_enforces_per_row_action_contract(self):
        trainer = object.__new__(WeightedActionTrainer)
        trainer._action_ids = (7,)
        trainer.action_weight = 20.0
        trainer.state = SimpleNamespace(global_step=1)
        trainer.args = SimpleNamespace(logging_steps=5)
        trainer.is_world_process_zero = lambda: False

        class DummyModel:
            def __call__(self, input_ids):
                batch, length = input_ids.shape
                return SimpleNamespace(logits=torch.zeros(batch, length, 16))

        base_inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "labels": torch.tensor([[-100, 2, 7], [-100, 2, 3]]),
            "requires_action": torch.tensor([True, False]),
        }
        loss = trainer.compute_loss(DummyModel(), base_inputs)
        self.assertTrue(torch.isfinite(loss))

        bad_gui = {key: value.clone() for key, value in base_inputs.items()}
        bad_gui["labels"][0, -1] = 3
        with self.assertRaises(ValueError):
            trainer.compute_loss(DummyModel(), bad_gui)

        bad_qa = {key: value.clone() for key, value in base_inputs.items()}
        bad_qa["labels"][1, -1] = 7
        with self.assertRaises(ValueError):
            trainer.compute_loss(DummyModel(), bad_qa)

    def test_screenspot_pro_is_resized_before_stage_three_supervision(self):
        from PIL import Image

        from src.train.sft3 import ScreenSpotProSFTDataset

        dataset = object.__new__(ScreenSpotProSFTDataset)
        feature = dataset._process_sample(
            {
                "image": Image.new("RGB", (2000, 1000), "white"),
                "bbox": [0.4, 0.4, 0.6, 0.6],
                "instruction": "Click the centered target",
            }
        )
        self.assertEqual(feature["image"].size, (1600, 800))
        reasoning, action = extract_action_response(
            feature["messages"][-1]["content"]
        )
        self.assertTrue(reasoning)
        self.assertIn(action, ACTION_TOKENS)
        self.assertTrue(feature["requires_action"])

    def test_all_source_modules_import(self):
        import src

        modules = [module.name for module in pkgutil.walk_packages(src.__path__, "src.")]
        for module in modules:
            importlib.import_module(module)
        self.assertGreater(len(modules), 0)

    def test_global_batch_is_preserved(self):
        old_world_size = os.environ.get("WORLD_SIZE")
        try:
            for world_size, expected_accumulation in ((1, 8), (2, 4), (4, 2), (8, 1)):
                os.environ["WORLD_SIZE"] = str(world_size)
                accumulation, actual_world_size, global_batch = resolve_gradient_accumulation(1)
                self.assertEqual(accumulation, expected_accumulation)
                self.assertEqual(actual_world_size, world_size)
                self.assertEqual(global_batch, 8)
        finally:
            if old_world_size is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = old_world_size

    def test_downloaded_base_requires_phase_zero(self):
        model_path = Path("model/Qwen3-VL-4B-Instruct")
        if not model_path.exists():
            self.skipTest("downloaded base model is unavailable")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with self.assertRaises(ValueError):
            atomic_action_token_ids(tokenizer)
        tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
        token_ids = atomic_action_token_ids(tokenizer)
        self.assertEqual(len(token_ids), len(ACTION_TOKENS))
        self.assertEqual(len(set(token_ids.values())), len(ACTION_TOKENS))

    def test_stage_two_collator_has_system_role_and_one_action_label(self):
        model_path = Path("model/Qwen3-VL-4B-Instruct")
        if not model_path.exists():
            self.skipTest("downloaded base model is unavailable")
        from PIL import Image
        from transformers import AutoProcessor

        from src.train.sft2 import SFTDataCollator
        from src.utils.prompts import AGENT_SYSTEM_PROMPT

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
        action_ids = set(atomic_action_token_ids(processor.tokenizer).values())
        response = format_action_response("The target is to the right.", "<MOVE_RIGHT_CLO>")
        feature = {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": "[Action] Click the target"},
                {"role": "assistant", "content": response},
            ],
            "image": Image.new("RGB", (128, 128), "white"),
            "requires_action": True,
        }
        batch = SFTDataCollator(processor)([feature])
        self.assertEqual(batch["requires_action"].tolist(), [True])
        labeled_ids = batch["labels"][0][batch["labels"][0].ne(-100)].tolist()
        self.assertEqual(sum(token_id in action_ids for token_id in labeled_ids), 1)
        rendered = processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        self.assertIn("<|im_start|>system", rendered)

        qa_feature = {
            "messages": [
                {"role": "system", "content": "Answer accurately."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},
            ],
            "image": Image.new("RGB", (64, 64), "black"),
            "requires_action": False,
        }
        qa_batch = SFTDataCollator(processor)([qa_feature])
        qa_labels = qa_batch["labels"][0][qa_batch["labels"][0].ne(-100)].tolist()
        self.assertEqual(qa_batch["requires_action"].tolist(), [False])
        self.assertFalse(any(token_id in action_ids for token_id in qa_labels))

    def test_generation_stop_ignores_action_tokens_inside_think(self):
        model_path = Path("model/Qwen3-VL-4B-Instruct")
        if not model_path.exists():
            self.skipTest("downloaded base model is unavailable")
        import torch
        from transformers import AutoTokenizer

        from src.tools.runner import ActionStoppingCriteria

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
        action_id = atomic_action_token_ids(tokenizer)["<CLICK_SHORT>"]
        criterion = ActionStoppingCriteria(tokenizer, [action_id])

        prompt_ids = tokenizer.encode("system prompt containing </think>", return_tensors="pt")
        criterion.reset(prompt_ids.shape[1])
        inside_think = tokenizer.encode("<think>consider <CLICK_SHORT>", add_special_tokens=False)
        sequence = torch.cat([prompt_ids, torch.tensor([inside_think])], dim=1)
        self.assertFalse(criterion(sequence, scores=torch.empty(0)))

        complete = tokenizer.encode(
            "<think>consider</think>\n<CLICK_SHORT>", add_special_tokens=False
        )
        sequence = torch.cat([prompt_ids, torch.tensor([complete])], dim=1)
        self.assertTrue(criterion(sequence, scores=torch.empty(0)))


if __name__ == "__main__":
    unittest.main()
