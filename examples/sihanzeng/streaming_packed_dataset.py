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


class StreamingPackedDataset(IterableDataset):
    """Streaming dataset that blends, tokenizes, packs, and shuffles on-the-fly.

    Yields pre-packed dicts with keys:
        input_ids, labels, position_ids, seq_lens, seq_lens_padded
    all sized to packed_sequence_size, ready for packed_sequence_thd_collater.
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
        self.pack_size = packed_sequence_size or seq_length
        self.seed = seed
        self.message_key = message_key
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last
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

    def _convert_and_tokenize(self, sample):
        """Convert dialog to (input_ids, labels) with assistant-only loss masking.

        Uses keep_loss field from data if available, otherwise masks non-assistant turns.
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

            return input_ids, labels
        except Exception:
            return None

    def _tokenized_stream(self):
        """Yields (input_ids, labels) tuples from the blended source."""
        for sample in self.blended:
            result = self._convert_and_tokenize(sample)
            if result is not None:
                yield result

    def _packing_stream(self):
        """Online sequence packer. Yields fixed-size packed samples (THD format)."""
        pack_size = self.pack_size
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        buf_ids: list[int] = []
        buf_labels: list[int] = []
        buf_pos: list[int] = []
        buf_seq_lens: list[int] = []

        for input_ids, labels in self._tokenized_stream():
            seq_len = len(input_ids)

            space_left = pack_size - len(buf_ids)

            if seq_len <= space_left:
                buf_ids.extend(input_ids)
                buf_labels.extend(labels)
                buf_pos.extend(range(seq_len))
                buf_seq_lens.append(seq_len)
            else:
                if buf_ids:
                    yield self._finalize_pack(buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id)

                buf_ids = list(input_ids)
                buf_labels = list(labels)
                buf_pos = list(range(seq_len))
                buf_seq_lens = [seq_len]

        if buf_ids and not self.drop_last:
            yield self._finalize_pack(buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id)

    def _finalize_pack(self, buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id):
        """Pad to pack_size and return a THD-format dict."""
        pack_size = self.pack_size
        cur_len = len(buf_ids)
        pad_len = pack_size - cur_len

        if pad_len > 0:
            buf_ids = buf_ids + [pad_id] * pad_len
            buf_labels = buf_labels + [CROSS_ENTROPY_IGNORE_IDX] * pad_len
            last_pos = buf_pos[-1] if buf_pos else 0
            buf_pos = buf_pos + list(range(last_pos + 1, last_pos + 1 + pad_len))

        seq_lens_padded = list(buf_seq_lens)
        if pad_len > 0:
            seq_lens_padded[-1] = seq_lens_padded[-1] + pad_len

        return {
            "input_ids": buf_ids,
            "labels": buf_labels,
            "position_ids": buf_pos,
            "seq_lens": list(buf_seq_lens),
            "seq_lens_padded": seq_lens_padded,
        }

    def __iter__(self):
        """Yields packed samples with optional reservoir shuffle."""
        self._drop_count = 0
        self._total_count = 0
        packing_iter = self._packing_stream()

        if self.shuffle_buffer_size <= 0:
            yield from packing_iter
            return

        rng = random.Random(self.seed)
        buffer = []

        for item in packing_iter:
            buffer.append(item)
            if len(buffer) == self.shuffle_buffer_size:
                break

        if not buffer:
            return

        rng.shuffle(buffer)
        for item in packing_iter:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item

        rng.shuffle(buffer)
        yield from buffer
