"""Streaming unpacked dataset for Gemma4 SFT with native thinking format.

Converts <think>...</think> tags to Gemma4's native <|channel>thought format.
Streams JSONL from multiple sources with weighted sampling.
Yields individual conversations (unpacked) with proper loss masking.
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

    Uses incremental apply_chat_template to find turn boundaries.
    Returns (input_ids, shifted_labels) or None if too long/short.
    """
    try:
        input_ids = []
        labels = []

        for i in range(len(messages)):
            prefix = tokenizer.apply_chat_template(
                messages[:i], tokenize=False, add_generation_prompt=False,
            ) if i > 0 else ""
            full = tokenizer.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False,
            )

            prefix_ids = tokenizer(
                prefix, truncation=False, padding=False, return_tensors=None,
            )["input_ids"] if prefix else []
            full_ids = tokenizer(
                full, truncation=False, padding=False, return_tensors=None,
            )["input_ids"]

            turn_ids = full_ids[len(prefix_ids):]

            if loss_flags[i]:
                turn_labels = list(turn_ids)
            else:
                turn_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)

        if len(input_ids) > seq_length or len(input_ids) < 2:
            return None

        # Shift labels: labels[i] = input_ids[i+1] for loss positions
        shifted = [CROSS_ENTROPY_IGNORE_IDX] * len(labels)
        for j in range(len(labels) - 1):
            if labels[j] != CROSS_ENTROPY_IGNORE_IDX:
                shifted[j] = input_ids[j + 1]

        return input_ids, shifted
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class StreamingGemma4Dataset(IterableDataset):
    """Streaming dataset for Gemma4 SFT with native thinking format.

    - Converts <think>...</think> → <|channel>thought\\n...\\n<channel|>
    - Streams JSONL from multiple weighted sources
    - Yields individual conversations (unpacked)
    - Proper loss masking (assistant-only) with shifted labels
    - Reservoir shuffle for randomization
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer=None,
        *,
        seq_length: int = 8192,
        seed: int = 42,
        shuffle_buffer_size: int = 256,
        convert_thinking: bool = True,
        add_empty_thinking: bool = False,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
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
        # Expose for split_dataset_by_node sharding in build_dataloader
        self.dataset = self.blended

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

    def __iter__(self):
        self._drop_count = 0
        self._total_count = 0

        def sample_stream():
            for sample in self.blended:
                result = self._process_sample(sample)
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
