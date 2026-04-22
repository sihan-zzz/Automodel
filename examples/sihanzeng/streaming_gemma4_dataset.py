"""Streaming dataset for Gemma4 SFT/CPT using amaia-style data pipeline.

Uses composable operators from amaia_data_pipeline.py:
  from_jsonl_partitioned → repeat → map(tokenize) → pack → shuffle

Supports two modes:
  - SFT: dialog → tokenize with loss mask → pack(wrap=False, pad) → shuffle
  - CPT: text → tokenize all tokens → pack(wrap=True) → shuffle

Converts <think>...</think> tags to Gemma4's native <|channel>thought format.
Streams JSONL from multiple weighted sources with file-level sharding.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from torch.utils.data import IterableDataset

from amaia_data_pipeline import StreamDataset, CROSS_ENTROPY_IGNORE_IDX

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def _parse_dialog(sample: dict) -> tuple[list[dict], list[bool]] | None:
    """Parse amaia/OpenAI dialog format into (messages, should_keep_loss)."""
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
    """Convert <think>...</think> to Gemma4 native <|channel>thought format."""
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
    """Convert messages to use Gemma4 native thinking format."""
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
    """Tokenize messages with per-turn loss masking and label shifting."""
    try:
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            add_generation_prompt=False,
        )
        input_ids = list(tokenized["input_ids"])

        if len(input_ids) > seq_length or len(input_ids) < 2:
            return None

        assistant_mask = [0] * len(input_ids)
        for i in range(len(messages)):
            if not loss_flags[i]:
                continue

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
    """Streaming dataset for Gemma4 SFT/CPT using amaia-style pipeline.

    Pipeline:
      for each source:
        from_jsonl_partitioned(path, dp_rank, dp_world_size)
          .repeat(n=max_epochs)          # None = infinite
          .map(tokenize_fn)              # returns None to skip
      across sources:
        StreamDataset.sample(datasets, weights, seed)
      then:
        .pack_sft(pack_size) or .pack_cpt(pack_size) or identity
        .shuffle(buffer_size, seed)

    Modes:
      - "sft": parse dialog, tokenize with loss mask, pack without wrap
      - "cpt": extract text field, tokenize all tokens, pack with wrap
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer=None,
        *,
        mode: str = "sft",
        seq_length: int = 8192,
        pack_size: int = 0,
        seed: int = 42,
        shuffle_buffer_size: int = 64,
        max_epochs: int | None = None,
        convert_thinking: bool = True,
        add_empty_thinking: bool = False,
        text_field: str = "text",
        split: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.mode = mode
        self.seq_length = seq_length
        self.pack_size = pack_size
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_epochs = max_epochs
        self.convert_thinking = convert_thinking
        self.add_empty_thinking = add_empty_thinking
        self.text_field = text_field
        self.sources = sources
        self._drop_count = 0
        self._total_count = 0

    @staticmethod
    def _get_dp_info():
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return 0, 1

    def _process_sft(self, sample: dict) -> dict | None:
        """Process one SFT sample: dialog → tokenize → {input_ids, labels}."""
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

        input_ids, labels = result
        return {"input_ids": input_ids, "labels": labels}

    def _process_cpt(self, sample: dict) -> dict | None:
        """Process one CPT sample: text → tokenize → {input_ids, labels}."""
        text = sample.get(self.text_field, "")
        if not text:
            return None

        try:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None

        if len(input_ids) < 2:
            return None

        # CPT: all tokens are training targets (shifted)
        # input_ids[:-1] → labels = input_ids[1:]
        labels = list(input_ids[1:])
        input_ids = list(input_ids[:-1])
        return {"input_ids": input_ids, "labels": labels}

    def _build_pipeline(self) -> StreamDataset:
        """Build the amaia-style composable pipeline."""
        dp_rank, dp_world_size = self._get_dp_info()
        process_fn = self._process_sft if self.mode == "sft" else self._process_cpt

        per_source_datasets = []
        weights = []
        for src in self.sources:
            path = src["path"]
            weight = float(src.get("weight", 1.0))
            max_epochs = src.get("max_epochs", self.max_epochs)
            logger.info(
                f"Building pipeline: {path} (weight={weight}, "
                f"max_epochs={max_epochs}, dp_rank={dp_rank}/{dp_world_size})"
            )

            ds = (
                StreamDataset.from_jsonl_partitioned(path, dp_rank, dp_world_size)
                .repeat(n=max_epochs)
                .map(process_fn)
            )
            per_source_datasets.append(ds)
            weights.append(weight)

        if len(per_source_datasets) == 1:
            pipeline = per_source_datasets[0]
        else:
            pipeline = StreamDataset.sample(
                per_source_datasets, weights, seed=self.seed,
            )

        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        if self.pack_size > 0:
            if self.mode == "sft":
                pipeline = pipeline.pack_sft(
                    self.pack_size, pad_id=pad_id, lookahead=64,
                )
            else:
                pipeline = pipeline.pack_cpt(self.pack_size)

        if self.shuffle_buffer_size > 0:
            pipeline = pipeline.shuffle(
                buffer_size=self.shuffle_buffer_size, seed=self.seed,
            )

        return pipeline

    def __iter__(self):
        self._drop_count = 0
        self._total_count = 0
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        pipeline = self._build_pipeline()

        for item in pipeline:
            if self.pack_size > 0:
                yield item
            else:
                yield {
                    "input_ids": item["input_ids"],
                    "labels": item["labels"],
                    "attention_mask": [1] * len(item["input_ids"]),
                    "___PAD_TOKEN_IDS___": {
                        "input_ids": pad_token_id,
                        "labels": -100,
                        "attention_mask": 0,
                    },
                }
