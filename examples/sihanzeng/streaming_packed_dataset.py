"""Streaming packed blended chat dataset for NeMo Automodel.

amaia-style zero-latency data pipeline:
- Streams JSONL from multiple sources with weighted sampling (HF interleave_datasets)
- Packs multiple tokenized conversations into fixed-length windows on-the-fly
- Reservoir shuffle for approximate randomization with bounded memory
- Outputs THD-format packed samples compatible with packed_sequence_thd_collater
- Loss masking: only assistant turns contribute to loss (user/system masked to -100)
- Samples exceeding max length are dropped, not truncated

No upfront data loading. No offline packing pass. Near-zero startup latency.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import torch
from datasets import VerificationMode, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100


def packed_thd_collater_with_mm(batch):
    """Wraps packed_sequence_thd_collater and adds mm_token_type_ids (required by Gemma4)."""
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    result = packed_sequence_thd_collater(batch)
    if "input_ids" in result:
        result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"])
    return result


class StreamingUnpackedDataset(IterableDataset):
    """Streaming dataset that blends, tokenizes, and shuffles on-the-fly.

    Unlike StreamingPackedDataset, yields individual conversations (not packed).
    Each sample is a single tokenized conversation with proper loss masking.
    Compatible with default_collater (pads within batch).
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer=None,
        *,
        seq_length: int = 8192,
        packed_sequence_size: Optional[int] = None,
        seed: int = 42,
        message_key: str = "dialog",
        shuffle_buffer_size: int = 64,
        drop_last: bool = True,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        self.message_key = message_key
        self.shuffle_buffer_size = shuffle_buffer_size
        self._drop_count = 0
        self._total_count = 0

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
        self.dataset = self.blended

    def _convert_and_tokenize(self, sample):
        """Convert dialog to (input_ids, labels) using apply_chat_template.

        Uses the tokenizer's chat template for correct formatting.
        Labels use next-token prediction: labels[i] = input_ids[i+1] for
        assistant turns, -100 for user/system turns.
        Returns None if sample exceeds seq_length (drop, not truncate).
        """
        dialog = sample.get("dialog", sample.get("conversations", []))
        keep_loss = sample.get("keep_loss", None)

        messages = []
        should_keep_loss = []
        for i, turn in enumerate(dialog):
            if isinstance(turn, dict):
                role = turn.get("source", turn.get("role", turn.get("from", "user")))
                content = turn.get("body", turn.get("content", turn.get("value", "")))
                role_map = {
                    "human": "user", "gpt": "assistant", "model": "assistant",
                    "User": "user", "Assistant": "assistant", "System": "system",
                }
                role = role_map.get(role, role.lower())
                if role in ("user", "assistant", "system") and content:
                    messages.append({"role": role, "content": content})
                    if keep_loss is not None and i < len(keep_loss):
                        should_keep_loss.append(bool(keep_loss[i]))
                    else:
                        should_keep_loss.append(role == "assistant")

        if len(messages) < 2:
            return None

        try:
            # Tokenize incrementally to find turn boundaries
            input_ids = []
            labels = []

            for i in range(len(messages)):
                prefix = self.tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=False,
                ) if i > 0 else ""
                full = self.tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=False, add_generation_prompt=False,
                )

                prefix_ids = self.tokenizer(
                    prefix, truncation=False, padding=False, return_tensors=None,
                )["input_ids"] if prefix else []
                full_ids = self.tokenizer(
                    full, truncation=False, padding=False, return_tensors=None,
                )["input_ids"]

                turn_ids = full_ids[len(prefix_ids):]

                if should_keep_loss[i]:
                    turn_labels = list(turn_ids)
                else:
                    turn_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(turn_ids)

                input_ids.extend(turn_ids)
                labels.extend(turn_labels)

            self._total_count += 1

            if len(input_ids) > self.seq_length:
                self._drop_count += 1
                if self._drop_count % 1000 == 1:
                    logger.info(
                        f"Dropped {self._drop_count}/{self._total_count} samples "
                        f"exceeding {self.seq_length} tokens (this one: {len(input_ids)})"
                    )
                return None

            if len(input_ids) < 2:
                return None

            # Shift labels: labels[i] = input_ids[i+1] for loss positions
            shifted = [CROSS_ENTROPY_IGNORE_IDX] * len(labels)
            for j in range(len(labels) - 1):
                if labels[j] != CROSS_ENTROPY_IGNORE_IDX:
                    shifted[j] = input_ids[j + 1]

            return input_ids, shifted
        except Exception:
            return None

    def __iter__(self):
        """Yields individual tokenized conversations with reservoir shuffle."""
        self._drop_count = 0
        self._total_count = 0

        def sample_stream():
            for sample in self.blended:
                result = self._convert_and_tokenize(sample)
                if result is not None:
                    input_ids, labels = result
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "mm_token_type_ids": [0] * len(input_ids),
                    }

        stream = sample_stream()

        if self.shuffle_buffer_size <= 0:
            yield from stream
            return

        rng = random.Random(self.seed)
        buffer = []

        for item in stream:
            buffer.append(item)
            if len(buffer) == self.shuffle_buffer_size:
                break

        if not buffer:
            return

        rng.shuffle(buffer)
        for item in stream:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item

        rng.shuffle(buffer)
        yield from buffer
