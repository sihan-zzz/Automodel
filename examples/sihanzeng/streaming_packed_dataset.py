"""Streaming packed blended chat dataset for NeMo Automodel.

amaia-style zero-latency data pipeline:
- Streams JSONL from multiple sources with weighted sampling (HF interleave_datasets)
- Packs multiple tokenized conversations into fixed-length windows on-the-fly
- Reservoir shuffle for approximate randomization with bounded memory
- Outputs THD-format packed samples compatible with packed_sequence_thd_collater

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
        """Convert dialog format to messages and tokenize. Returns (input_ids, labels) or None."""
        dialog = sample.get("dialog", sample.get("conversations", []))
        messages = []
        for turn in dialog:
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

        if len(messages) < 2:
            return None

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            encoded = self.tokenizer(
                text, truncation=True, max_length=self.seq_length,
                padding=False, return_tensors=None,
            )
            input_ids = encoded["input_ids"]
            if len(input_ids) < 2:
                return None
            return input_ids
        except Exception:
            return None

    def _tokenized_stream(self):
        """Yields tokenized sequences (list[int]) from the blended source."""
        for sample in self.blended:
            input_ids = self._convert_and_tokenize(sample)
            if input_ids is not None:
                yield input_ids

    def _packing_stream(self):
        """Online sequence packer. Yields fixed-size packed samples (THD format).

        Greedy bin-packing: fills a buffer until pack_size, then emits.
        Sequences that don't fit are split at the boundary (no wrapping across
        document boundaries for SFT — the remainder starts fresh in the next pack).
        """
        pack_size = self.pack_size
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        # Current pack state
        buf_ids: list[int] = []
        buf_labels: list[int] = []
        buf_pos: list[int] = []
        buf_seq_lens: list[int] = []
        cur_pos = 0

        for input_ids in self._tokenized_stream():
            seq_len = len(input_ids)

            if seq_len > pack_size:
                input_ids = input_ids[:pack_size]
                seq_len = pack_size

            space_left = pack_size - len(buf_ids)

            if seq_len <= space_left:
                buf_ids.extend(input_ids)
                buf_labels.extend(input_ids)
                buf_pos.extend(range(seq_len))
                buf_seq_lens.append(seq_len)
            else:
                # Emit current pack (pad the remaining space)
                if buf_ids:
                    yield self._finalize_pack(buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id)

                # Start fresh pack with this sequence
                buf_ids = list(input_ids)
                buf_labels = list(input_ids)
                buf_pos = list(range(seq_len))
                buf_seq_lens = [seq_len]

        # Emit final partial pack
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
        packing_iter = self._packing_stream()

        if self.shuffle_buffer_size <= 0:
            yield from packing_iter
            return

        # Reservoir shuffle over packed samples
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
