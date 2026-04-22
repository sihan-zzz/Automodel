"""Streaming dataset for Gemma4 SFT with native thinking format.

Supports both unpacked (one conversation per sample) and packed (multiple
conversations per fixed-length window) modes.

Converts <think>...</think> tags to Gemma4's native <|channel>thought format.
Streams JSONL from multiple sources with weighted sampling.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional

from datasets import VerificationMode, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

CROSS_ENTROPY_IGNORE_IDX = -100


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def _parse_dialog(sample: dict) -> tuple[list[dict], list[bool]] | None:
    """Parse amaia/OpenAI dialog format into (messages, should_keep_loss).

    Handles both amaia format (source/body) and OpenAI format (role/content).
    Returns None if the dialog has fewer than 2 valid turns.
    """
    dialog = sample.get("dialog", sample.get("conversations", sample.get("messages", [])))
    keep_loss = sample.get("keep_loss", None)

    messages = []
    loss_flags = []
    for i, turn in enumerate(dialog):
        if not isinstance(turn, dict):
            continue
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
                loss_flags.append(bool(keep_loss[i]))
            else:
                loss_flags.append(role == "assistant")

    if len(messages) < 2:
        return None
    return messages, loss_flags


def _convert_think_to_gemma4_native(content: str) -> str:
    """Convert <think>...</think> to Gemma4 native <|channel>thought format.

    Input:  <think>reasoning here</think>answer here
    Output: <|channel>thought\nreasoning here\n<channel|>answer here

    If no <think> tags, returns content unchanged.
    """
    if "<think>" not in content:
        return content

    def replace_think(match):
        thinking = match.group(1).strip()
        return f"<|channel>thought\n{thinking}\n<channel|>"

    return re.sub(r"<think>(.*?)</think>", replace_think, content, flags=re.DOTALL)


def convert_messages_for_gemma4(
    messages: list[dict],
    add_empty_thinking: bool = False,
) -> list[dict]:
    """Convert messages to use Gemma4 native thinking format.

    For assistant messages containing <think> tags, converts to
    <|channel>thought format. Other messages pass through unchanged.

    If add_empty_thinking=True, prepends an empty thinking block
    to assistant messages without <think> tags, preserving the
    model's thinking capability format.
    """
    converted = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "<think>" in content:
                content = _convert_think_to_gemma4_native(content)
            elif add_empty_thinking:
                content = f"<|channel>thought\n<channel|>{content}"
            converted.append({"role": msg["role"], "content": content})
        else:
            converted.append(msg)
    return converted


def tokenize_with_loss_mask(
    tokenizer,
    messages: list[dict],
    loss_flags: list[bool],
    seq_length: int,
) -> tuple[list[int], list[int]] | None:
    """Tokenize messages with per-turn loss masking and label shifting.

    Uses single apply_chat_template(tokenize=True) call matching ChatDataset,
    then builds assistant mask via incremental prefix matching.
    Returns (input_ids, shifted_labels) or None if too long/short.
    """
    try:
        # Single tokenize call — matches ChatDataset's format_chat_template
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            add_generation_prompt=False,
        )
        input_ids = list(tokenized["input_ids"])

        if len(input_ids) > seq_length or len(input_ids) < 2:
            return None

        # Build assistant mask via incremental prefix matching
        assistant_mask = [0] * len(input_ids)
        for i in range(len(messages)):
            if not loss_flags[i]:
                continue

            # Find where this turn starts/ends by comparing prefix tokenizations
            prefix_text = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=False,
            ) if i > 0 else ""
            full_text = tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False,
            )

            prefix_ids = tokenizer(
                prefix_text, truncation=False, padding=False, return_tensors=None,
            )["input_ids"] if prefix_text else []
            full_ids = tokenizer(
                full_text, truncation=False, padding=False, return_tensors=None,
            )["input_ids"]

            start = len(prefix_ids)
            end = len(full_ids)
            for j in range(start, min(end, len(input_ids))):
                assistant_mask[j] = 1

        # Apply mask and shift labels (matching ChatDataset's _package_tokenized_example)
        labels = list(input_ids)
        labels = [l if m else CROSS_ENTROPY_IGNORE_IDX for l, m in zip(labels, assistant_mask)]

        # Shift: input_ids = input_ids[:-1], labels = labels[1:]
        input_ids = input_ids[:-1]
        labels = labels[1:]

        if len(input_ids) < 2:
            return None

        return input_ids, labels
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class StreamingGemma4Dataset(IterableDataset):
    """Streaming dataset for Gemma4 SFT with native thinking format.

    - Converts <think>...</think> → <|channel>thought\\n...\\n<channel|>
    - Streams JSONL from multiple weighted sources
    - Supports unpacked (one conv per sample) or packed (multiple convs per window)
    - Proper loss masking (assistant-only) with shifted labels
    - amaia-style line-level sharding for even DP distribution
    - Reservoir shuffle for randomization

    When pack_size > 0, yields packed samples with seq_lens for block-diagonal
    attention masking (compatible with packed_thd_collater_with_mm).
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer=None,
        *,
        seq_length: int = 8192,
        pack_size: int = 0,
        seed: int = 42,
        shuffle_buffer_size: int = 256,
        convert_thinking: bool = True,
        add_empty_thinking: bool = False,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.pack_size = pack_size
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.convert_thinking = convert_thinking
        self.add_empty_thinking = add_empty_thinking
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
            ds = ds.shuffle(seed=seed, buffer_size=10000)
            datasets_list.append(ds)
            weights.append(weight)

        total_w = sum(weights)
        self.probabilities = [w / total_w for w in weights]
        logger.info(f"Blending {len(sources)} sources with probabilities: {self.probabilities}")
        logger.info(f"convert_thinking={convert_thinking}, seq_length={seq_length}")

        self.blended = interleave_datasets(
            datasets_list,
            probabilities=self.probabilities,
            seed=seed,
            stopping_strategy="all_exhausted",
        )
        # NOTE: we do NOT expose self.dataset for split_dataset_by_node.
        # Instead we implement amaia-style line-level interleaving in __iter__
        # via sample_idx % world_size == rank, which gives even distribution
        # regardless of file sizes. See _get_dp_info().
        self._datasets_list = datasets_list

    def _process_sample(self, sample):
        """Convert raw sample to (input_ids, labels) or None."""
        parsed = _parse_dialog(sample)
        if parsed is None:
            return None
        messages, loss_flags = parsed

        if self.convert_thinking or self.add_empty_thinking:
            messages = convert_messages_for_gemma4(
                messages, add_empty_thinking=self.add_empty_thinking,
            )

        self._total_count += 1
        result = tokenize_with_loss_mask(
            self.tokenizer, messages, loss_flags, self.seq_length,
        )
        if result is None:
            self._drop_count += 1
            if self._drop_count % 1000 == 1:
                logger.info(f"Dropped {self._drop_count}/{self._total_count} samples")
            return None
        return result

    @staticmethod
    def _get_dp_info():
        """Get DP rank and world size for line-level sharding."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return 0, 1

    def _tokenized_stream(self):
        """Yield (input_ids, labels) tuples with amaia-style line-level sharding."""
        dp_rank, dp_world_size = self._get_dp_info()
        for idx, sample in enumerate(self.blended):
            if idx % dp_world_size != dp_rank:
                continue
            result = self._process_sample(sample)
            if result is not None:
                yield result

    def _pack_stream(self, token_stream):
        """Online greedy packer: fill fixed-length windows from token stream.

        Yields dicts with input_ids, labels, position_ids, seq_lens, seq_lens_padded
        compatible with packed_thd_collater_with_mm + SDPA block-diagonal mask.
        """
        pack_size = self.pack_size
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        buf_ids, buf_labels, buf_pos, buf_seq_lens = [], [], [], []

        for input_ids, labels in token_stream:
            seq_len = len(input_ids)
            if seq_len > pack_size:
                continue

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

        if buf_ids:
            yield self._finalize_pack(buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id)

    def _finalize_pack(self, buf_ids, buf_labels, buf_pos, buf_seq_lens, pad_id):
        pack_size = self.pack_size
        pad_len = pack_size - len(buf_ids)

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
        self._drop_count = 0
        self._total_count = 0
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        token_stream = self._tokenized_stream()

        if self.pack_size > 0:
            raw_stream = self._pack_stream(token_stream)
        else:
            def unpacked_stream():
                for input_ids, labels in token_stream:
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": [1] * len(input_ids),
                        "___PAD_TOKEN_IDS___": {
                            "input_ids": pad_token_id,
                            "labels": -100,
                            "attention_mask": 0,
                        },
                    }
            raw_stream = unpacked_stream()

        # Reservoir shuffle
        if self.shuffle_buffer_size <= 0:
            yield from raw_stream
            return

        rng = random.Random(self.seed)
        buffer = []
        for item in raw_stream:
            buffer.append(item)
            if len(buffer) == self.shuffle_buffer_size:
                break
        if not buffer:
            return

        rng.shuffle(buffer)
        for item in raw_stream:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item

        rng.shuffle(buffer)
        yield from buffer
