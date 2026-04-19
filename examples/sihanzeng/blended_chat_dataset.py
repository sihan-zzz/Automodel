"""Streaming weighted blended chat dataset for NeMo Automodel.
Reads multiple JSONL sources with per-dataset weights using HF interleave_datasets.
No upfront data loading — streams directly from disk.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from datasets import load_dataset, interleave_datasets, VerificationMode
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class BlendedChatDataset(IterableDataset):
    """Streaming chat dataset that blends multiple JSONL sources with weights.

    Config example:
        dataset:
          _target_: blended_chat_dataset.BlendedChatDataset
          sources:
            - path: /data/tulu3_sft
              weight: 0.8
            - path: /data/step_3p5_flash_sft
              weight: 0.2
          seq_length: 8192
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer=None,
        *,
        seq_length: Optional[int] = 8192,
        seed: int = 42,
        message_key: str = "dialog",
        split: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        self.message_key = message_key

        datasets_list = []
        weights = []
        for src in sources:
            path = src["path"]
            weight = float(src.get("weight", 1.0))
            logger.info(f"Loading streaming dataset: {path} (weight={weight})")
            ds = load_dataset(
                "json",
                data_files=f"{path}/**/*.jsonl",
                split="train",
                streaming=True,
                verification_mode=VerificationMode.NO_CHECKS,
            )
            datasets_list.append(ds)
            weights.append(weight)

        total_w = sum(weights)
        self.probabilities = [w / total_w for w in weights]
        logger.info(f"Blending with probabilities: {self.probabilities}")

        self.blended = interleave_datasets(
            datasets_list,
            probabilities=self.probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )

    def _convert_sample(self, sample):
        """Convert our dialog format to OpenAI messages format."""
        dialog = sample.get("dialog", sample.get("conversations", []))
        messages = []
        for turn in dialog:
            if isinstance(turn, dict):
                role = turn.get("source", turn.get("role", turn.get("from", "user")))
                content = turn.get("body", turn.get("content", turn.get("value", "")))
                role_map = {"human": "user", "gpt": "assistant", "model": "assistant",
                            "User": "user", "Assistant": "assistant", "System": "system"}
                role = role_map.get(role, role.lower())
                if role in ("user", "assistant", "system") and content:
                    messages.append({"role": role, "content": content})
        return messages

    def __iter__(self):
        for sample in self.blended:
            messages = self._convert_sample(sample)
            if len(messages) < 2:
                continue

            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.seq_length,
                    padding=False,
                    return_tensors=None,
                )

                input_ids = encoded["input_ids"]
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.copy(),
                    "attention_mask": encoded.get("attention_mask", [1] * len(input_ids)),
                    "mm_token_type_ids": [0] * len(input_ids),
                }
            except Exception as e:
                logger.debug(f"Skipping sample: {e}")
                continue
