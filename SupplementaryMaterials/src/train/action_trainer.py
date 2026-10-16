"""Strict action-token validation and weighted SFT loss."""

from __future__ import annotations

import os

import torch
from transformers import Trainer

from ..utils.action import atomic_action_token_ids
from ..utils.parameters import HYPERPARAMS as HP


def resolve_gradient_accumulation(per_device_batch_size: int) -> tuple[int, int, int]:
    """Preserve the configured global batch size across DDP world sizes."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    target_global_batch = HP.SFT_GLOBAL_BATCH_SIZE
    per_update = per_device_batch_size * world_size
    if target_global_batch % per_update != 0:
        raise ValueError(
            f"global batch {target_global_batch} is not divisible by "
            f"per_device_batch ({per_device_batch_size}) * world_size ({world_size})"
        )
    accumulation = target_global_batch // per_update
    if accumulation < 1:
        raise ValueError(
            f"world size {world_size} makes the minimum global batch {per_update}, "
            f"larger than configured {target_global_batch}"
        )
    return accumulation, world_size, per_update * accumulation


class WeightedActionTrainer(Trainer):
    """Weight GUI actions while retaining ordinary LM loss for QA rows."""

    def __init__(self, *args, action_weight: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        processor = getattr(self, "processing_class", None)
        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer is None:
            raise ValueError("WeightedActionTrainer requires a tokenizer/processor")
        self.action_token_ids = atomic_action_token_ids(tokenizer)
        self.action_weight = HP.ACTION_TOKEN_WEIGHT if action_weight is None else action_weight
        self._action_ids = tuple(self.action_token_ids.values())
        print(
            f"[Action Loss] Validated {len(self._action_ids)} atomic action tokens; "
            f"weight={self.action_weight:g}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        requires_action = inputs.get("requires_action")
        if requires_action is None:
            raise ValueError("collator must provide requires_action for every sample")
        requires_action = requires_action.to(device=labels.device, dtype=torch.bool)
        model_inputs = {
            name: value
            for name, value in inputs.items()
            if name not in {"labels", "requires_action"}
        }
        outputs = model(**model_inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        raw_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)

        action_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for token_id in self._action_ids:
            action_mask |= shift_labels.eq(token_id)
        action_counts = action_mask.sum(dim=1)
        expected_counts = requires_action.to(dtype=action_counts.dtype)
        if not torch.equal(action_counts, expected_counts):
            counts = action_counts.detach().cpu().tolist()
            expected = expected_counts.detach().cpu().tolist()
            raise ValueError(
                "GUI rows must contain exactly one labeled atomic action token and QA "
                f"rows must contain none; batch counts={counts}, expected={expected}"
            )

        valid_mask = shift_labels.ne(-100)
        weights = torch.ones_like(raw_loss)
        weights.masked_fill_(action_mask, self.action_weight)
        weights *= valid_mask
        loss = (raw_loss * weights).sum() / weights.sum().clamp_min(1.0)

        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():
            with torch.no_grad():
                action_loss = raw_loss[action_mask].mean().item() if action_mask.any() else 0.0
                language_mask = valid_mask & ~action_mask
                language_loss = (
                    raw_loss[language_mask].mean().item() if language_mask.any() else 0.0
                )
                print(
                    f"\n[Step {self.state.global_step}] weighted={loss.item():.4f} | "
                    f"action={action_loss:.4f} | language={language_loss:.4f}"
                )

        return (loss, outputs) if return_outputs else loss
