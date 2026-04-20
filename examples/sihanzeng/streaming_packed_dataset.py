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

    def _build_gemma4_text(self, turns, roles):
        """Build Gemma4 chat format manually to preserve <think> tags.

        Gemma4 format: <bos><|turn>role\ncontent<turn|>\n...
        The native apply_chat_template strips <think> blocks via strip_thinking(),
        so we construct the format directly.
        """
        role_map_gemma = {"assistant": "model", "user": "user", "system": "system"}
        parts = []
        for i, (turn, role) in enumerate(zip(turns, roles)):
            gemma_role = role_map_gemma.get(role, role)
            parts.append(f"<|turn>{gemma_role}\n{turn}<turn|>\n")
        return "".join(parts)

    def _convert_and_tokenize(self, sample):
        """Convert dialog to (input_ids, labels) with assistant-only loss masking.

        Uses keep_loss field from data if available, otherwise masks non-assistant turns.
        Preserves <think>...</think> tags in assistant content (Gemma4's template strips them).
        Returns None if sample exceeds seq_length (drop, not truncate).
        """
        dialog = sample.get("dialog", sample.get("conversations", []))
        keep_loss = sample.get("keep_loss", None)

        turns = []
        roles = []
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
                    turns.append(content)
                    roles.append(role)
                    if keep_loss is not None and i < len(keep_loss):
                        should_keep_loss.append(bool(keep_loss[i]))
                    else:
                        should_keep_loss.append(role == "assistant")

        if len(turns) < 2:
            return None

        try:
            input_ids = []
            labels = []

            # Tokenize BOS
            bos_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
            input_ids.extend(bos_ids)
            labels.extend([CROSS_ENTROPY_IGNORE_IDX] * len(bos_ids))

            for i, (content, role, has_loss) in enumerate(zip(turns, roles, should_keep_loss)):
                gemma_role = "model" if role == "assistant" else role
                header = f"<|turn>{gemma_role}\n"
                footer = "<turn|>\n"

                header_ids = self.tokenizer(
                    header, add_special_tokens=False, return_tensors=None,
                )["input_ids"]
                content_ids = self.tokenizer(
                    content, add_special_tokens=False, return_tensors=None,
                )["input_ids"]
                footer_ids = self.tokenizer(
                    footer, add_special_tokens=False, return_tensors=None,
                )["input_ids"]

                # Header (turn marker + role): always masked
                input_ids.extend(header_ids)
                labels.extend([CROSS_ENTROPY_IGNORE_IDX] * len(header_ids))

                # Content: only compute loss if has_loss
                input_ids.extend(content_ids)
                if has_loss:
                    labels.extend(list(content_ids))
                else:
                    labels.extend([CROSS_ENTROPY_IGNORE_IDX] * len(content_ids))

                # Footer (turn end): masked
                input_ids.extend(footer_ids)
                labels.extend([CROSS_ENTROPY_IGNORE_IDX] * len(footer_ids))

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

            return input_ids, labels
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
