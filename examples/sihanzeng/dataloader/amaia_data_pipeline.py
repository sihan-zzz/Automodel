"""Standalone data pipeline operators adapted from amaia.

Replicates amaia's composable Dataset operator chain for streaming data loading:
  - OneFileLines: JSONL file reader with seek-based reset
  - Repeat: infinite cycling (or n epochs)
  - Shuffle: buffer shuffle with deterministic RNG
  - Partition: line-level idx % world_size == rank sharding
  - Mix + SamplePolicy: weighted multi-source sampling with exhaust handling
  - Chain: sequential concatenation of datasets
  - Map / Filter: functional transforms

Pipeline for SFT:
  from_jsonl_partitioned(path, rank, world_size)
    .repeat(n=None)
    .map(tokenize_fn)
    .filter(lambda x: x is not None)
    .shuffle(buffer_size=64, seed=0)

Multiple sources:
  StreamDataset.sample(datasets, weights, seed=0)
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class StreamDataset(IterableDataset[T], ABC):
    """Base class matching amaia's Dataset(Stateful, IterableDataset[T])."""

    @abstractmethod
    def reset(self) -> None: ...

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> T: ...

    def repeat(self, n: int | None = None) -> Repeat[T]:
        return Repeat(self, n=n)

    def shuffle(self, *, buffer_size: int = 64, seed: int = 0) -> Shuffle[T]:
        return Shuffle(self, buffer_size=buffer_size, seed=seed)

    def partition(self, rank_world_fn: Callable[[], tuple[int, int]]) -> Partition[T]:
        return Partition(self, rank_world_fn)

    def map(self, fn: Callable) -> Map:
        return Map(self, fn)

    def filter(self, fn: Callable[[T], bool]) -> Filter[T]:
        return Filter(self, fn)

    def batch(self, batch_size: int) -> Batch[T]:
        return Batch(self, batch_size)

    def pack_sft(self, pack_size: int, pad_id: int = 0, lookahead: int = 64) -> PackSFT:
        return PackSFT(self, pack_size, pad_id=pad_id, lookahead=lookahead)

    def pack_cpt(self, pack_size: int) -> PackCPT:
        return PackCPT(self, pack_size)

    @staticmethod
    def sample(
        datasets: list[StreamDataset[T]],
        weights: list[float] | None = None,
        *,
        seed: int = 0,
    ) -> Mix[T]:
        return Mix(datasets, policy=SamplePolicy(weights, seed=seed))

    @staticmethod
    def chain(datasets: list[StreamDataset[T]]) -> Mix[T]:
        return Mix(datasets, policy=ChainPolicy())

    @staticmethod
    def from_jsonl(path: str | Path) -> OneFileLines:
        return OneFileLines(path)

    @staticmethod
    def from_jsonl_partitioned(
        path: str | Path,
        rank: int,
        world_size: int,
    ) -> StreamDataset[dict]:
        p = Path(path)
        if p.is_dir():
            file_paths = sorted(p.glob("*.jsonl")) + sorted(p.glob("*.jsonl.gz"))
        elif p.is_file():
            file_paths = [p]
        else:
            raise FileNotFoundError(f"Path not found: {path}")

        assignments = file_lines_assignments(rank, world_size, len(file_paths))
        datasets: list[StreamDataset[dict]] = []
        for a in assignments:
            reader = OneFileLines(file_paths[a.file_idx])
            if a.file_world_size > 1:
                datasets.append(
                    reader.filter(
                        lambda item, a=a: item["_line_no"] % a.file_world_size == a.file_rank
                    )
                )
            else:
                datasets.append(reader)
        return StreamDataset.chain(datasets)


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

class OneFileLines(StreamDataset[dict]):
    """Read JSONL file line by line. Adapted from amaia's OneFileLines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.file = None
        self.pos = 0
        self.line_no = 0

        if not self.path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    def _close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def reset(self) -> None:
        self._close()
        self.pos = 0
        self.line_no = 0

    def __next__(self) -> dict:
        if self.file is None:
            self.file = open(self.path, "rb")
            self.file.seek(self.pos)

        line_bytes = self.file.readline()
        if not line_bytes:
            raise StopIteration

        prev_pos = self.pos
        self.pos += len(line_bytes)
        line_no = self.line_no
        self.line_no += 1

        text = line_bytes.decode("utf-8", errors="strict")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        data["_line_no"] = line_no
        data["_pos"] = prev_pos
        return data


# ---------------------------------------------------------------------------
# File-level partitioning (adapted from amaia)
# ---------------------------------------------------------------------------

@dataclass
class FileAssignment:
    file_idx: int
    file_rank: int = 0
    file_world_size: int = 1


def file_lines_assignments(
    rank: int, world_size: int, num_files: int
) -> list[FileAssignment]:
    if world_size == num_files:
        return [FileAssignment(file_idx=rank)]
    if world_size < num_files:
        base_files = num_files // world_size
        extra_files = num_files % world_size
        start_file = rank * base_files + min(rank, extra_files)
        end_file = start_file + base_files + (1 if rank < extra_files else 0)
        return [FileAssignment(file_idx) for file_idx in range(start_file, end_file)]
    # world_size > num_files: multiple ranks share files with line-level filtering
    base_workers = world_size // num_files
    extra_workers = world_size % num_files
    if rank < extra_workers * (base_workers + 1):
        file_idx = rank // (base_workers + 1)
        file_world_size = base_workers + 1
    else:
        file_idx = (
            rank - extra_workers * (base_workers + 1)
        ) // base_workers + extra_workers
        file_world_size = base_workers
    file_rank = rank % file_world_size
    return [
        FileAssignment(
            file_idx=file_idx, file_rank=file_rank, file_world_size=file_world_size
        )
    ]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class Repeat(StreamDataset[T]):
    """Repeat the wrapped dataset n times. None = forever."""

    def __init__(self, dataset: StreamDataset[T], n: int | None = None) -> None:
        self.dataset = dataset
        self.n = n
        self.i = 0
        self.empty = True
        self.samples_yielded = 0
        self.samples_per_epoch: int | None = None

    def reset(self) -> None:
        self.i = 0
        self.empty = True
        self.samples_yielded = 0
        self.samples_per_epoch = None
        self.dataset.reset()

    def __next__(self) -> T:
        while True:
            if self.n is not None and self.i >= self.n:
                raise StopIteration
            try:
                x = next(self.dataset)
                self.empty = False
                self.samples_yielded += 1
                return x
            except StopIteration:
                if self.empty:
                    raise
                if self.samples_per_epoch is None:
                    self.samples_per_epoch = self.samples_yielded
                logger.info(
                    f"Repeat: reset {self.i + 1}/{self.n} after {self.samples_yielded} samples"
                    f" (epoch {self.epoch_progress:.2f})"
                )
                self.dataset.reset()
                self.i += 1

    @property
    def epoch_progress(self) -> float:
        if self.samples_per_epoch and self.samples_per_epoch > 0:
            return self.samples_yielded / self.samples_per_epoch
        return float(self.i)


class Shuffle(StreamDataset[T]):
    """Buffer shuffle with deterministic torch.Generator RNG."""

    def __init__(
        self,
        dataset: StreamDataset[T],
        *,
        buffer_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.generator = torch.Generator().manual_seed(seed)
        self.buffer: list[T] = []

    def reset(self) -> None:
        self.dataset.reset()
        self.buffer = []

    def __next__(self) -> T:
        try:
            while len(self.buffer) < self.buffer_size:
                self.buffer.append(next(self.dataset))
        except StopIteration:
            pass

        if not self.buffer:
            raise StopIteration

        pop_idx = torch.randint(
            low=0, high=len(self.buffer), size=(1,), generator=self.generator
        )
        self.buffer[-1], self.buffer[pop_idx] = self.buffer[pop_idx], self.buffer[-1]
        return self.buffer.pop()


class Partition(StreamDataset[T]):
    """Line-level sharding: yield items where global_count % world_size == rank."""

    def __init__(
        self,
        dataset: StreamDataset[T],
        rank_world_fn: Callable[[], tuple[int, int]],
    ) -> None:
        self.dataset = dataset
        self.global_yield_count = -1
        self.rank_world_fn = rank_world_fn

    def reset(self) -> None:
        self.dataset.reset()
        self.global_yield_count = -1

    def __next__(self) -> T:
        rank, world_size = self.rank_world_fn()
        for item in self.dataset:
            self.global_yield_count += 1
            if self.global_yield_count % world_size == rank:
                return item
        raise StopIteration


class Map(StreamDataset):
    """Apply a function to each item. If fn returns None, skip the item."""

    def __init__(self, dataset: StreamDataset, fn: Callable) -> None:
        self.dataset = dataset
        self.fn = fn

    def reset(self) -> None:
        self.dataset.reset()

    def __next__(self):
        while True:
            item = next(self.dataset)
            result = self.fn(item)
            if result is not None:
                return result


class Filter(StreamDataset[T]):
    """Keep only items where fn(item) is True."""

    def __init__(self, dataset: StreamDataset[T], fn: Callable[[T], bool]) -> None:
        self.dataset = dataset
        self.fn = fn

    def reset(self) -> None:
        self.dataset.reset()

    def __next__(self) -> T:
        while True:
            item = next(self.dataset)
            if self.fn(item):
                return item


class Batch(StreamDataset[list[T]]):
    """Batch items. Drops remainder by default (like amaia)."""

    def __init__(
        self, dataset: StreamDataset[T], batch_size: int, *, yield_remainder: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.yield_remainder = yield_remainder

    def reset(self) -> None:
        self.dataset.reset()

    def __next__(self) -> list[T]:
        items = list(islice(self.dataset, self.batch_size))
        if items and (len(items) == self.batch_size or self.yield_remainder):
            return items
        raise StopIteration


# ---------------------------------------------------------------------------
# Multi-source mixing
# ---------------------------------------------------------------------------

class SamplePolicy:
    """Weighted random sampling across datasets. Adapted from amaia."""

    def __init__(
        self,
        weights: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self._init_weights = weights
        self.seed = seed
        self.generator = torch.Generator()
        self.num_datasets = 0
        self.active_datasets: list[int] = []

    def initialize(self, datasets: list) -> None:
        self.num_datasets = len(datasets)
        if self._init_weights is None:
            self._init_weights = [1.0] * self.num_datasets
        self.reset()

    def reset(self) -> None:
        self.generator.manual_seed(self.seed)
        self.active_datasets = list(range(self.num_datasets))

    def select_dataset(self) -> int | None:
        if not self.active_datasets:
            return None
        weights = torch.tensor(
            [self._init_weights[i] for i in self.active_datasets], dtype=torch.float
        )
        if weights.sum() == 0:
            return None
        active_idx = int(
            torch.multinomial(weights, 1, generator=self.generator).item()
        )
        return self.active_datasets[active_idx]

    def exhaust(self, idx: int) -> bool:
        self.active_datasets.remove(idx)
        return False


class ChainPolicy:
    """Sequential: exhaust dataset 0, then 1, then 2, etc."""

    def __init__(self) -> None:
        self.idx: int | None = 0
        self.cnt: int | None = None

    def initialize(self, datasets: list) -> None:
        self.cnt = len(datasets)

    def reset(self) -> None:
        self.idx = 0

    def select_dataset(self) -> int | None:
        if self.idx is not None and self.cnt is not None and self.idx >= self.cnt:
            self.idx = None
        return self.idx

    def exhaust(self, _: int) -> bool:
        assert self.idx is not None
        assert self.cnt is not None
        self.idx += 1
        if self.idx >= self.cnt:
            self.idx = None
        return self.idx is not None


class Mix(StreamDataset[T]):
    """Multi-source mixing with pluggable policy. Adapted from amaia."""

    def __init__(
        self,
        datasets: list[StreamDataset[T]],
        policy: SamplePolicy | ChainPolicy,
    ) -> None:
        self.datasets = datasets
        self.policy = policy
        self.policy.initialize(datasets)

    def reset(self) -> None:
        for ds in self.datasets:
            ds.reset()
        self.policy.reset()

    def __next__(self) -> T:
        while True:
            idx = self.policy.select_dataset()
            if idx is None:
                raise StopIteration
            try:
                return next(self.datasets[idx])
            except StopIteration:
                if self.policy.exhaust(idx):
                    self.datasets[idx].reset()


# ---------------------------------------------------------------------------
# Packing operators
# ---------------------------------------------------------------------------

CROSS_ENTROPY_IGNORE_IDX = -100


class PackSFT(StreamDataset[dict]):
    """Online SFT packer: no wrap, pad remainder, track seq_lens.

    Adapted from amaia's SequencePack with TokensPacker(wrap=False).
    Uses greedy bin-packing with a lookahead buffer for better utilization.

    Input items: {"input_ids": list[int], "labels": list[int]}
    Output items: {"input_ids": list, "labels": list, "position_ids": list,
                   "seq_lens": list, "seq_lens_padded": list}
    """

    def __init__(
        self,
        dataset: StreamDataset[dict],
        pack_size: int,
        pad_id: int = 0,
        lookahead: int = 64,
    ) -> None:
        self.dataset = dataset
        self.pack_size = pack_size
        self.pad_id = pad_id
        self.lookahead = lookahead
        self.pending: list[dict] = []
        self.buf_ids: list[int] = []
        self.buf_labels: list[int] = []
        self.buf_pos: list[int] = []
        self.buf_seq_lens: list[int] = []

    def reset(self) -> None:
        self.dataset.reset()
        self.pending = []
        self.buf_ids = []
        self.buf_labels = []
        self.buf_pos = []
        self.buf_seq_lens = []

    def _refill(self) -> None:
        while len(self.pending) < self.lookahead:
            try:
                item = next(self.dataset)
                sl = len(item["input_ids"])
                if sl <= self.pack_size:
                    self.pending.append(item)
            except StopIteration:
                break

    def _find_best_fit(self, space: int) -> int | None:
        best_idx, best_len = None, 0
        for i, item in enumerate(self.pending):
            sl = len(item["input_ids"])
            if sl <= space and sl > best_len:
                best_idx, best_len = i, sl
        return best_idx

    def _finalize(self) -> dict:
        pad_len = self.pack_size - len(self.buf_ids)
        ids = self.buf_ids
        labels = self.buf_labels
        pos = self.buf_pos
        seq_lens = list(self.buf_seq_lens)

        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            labels = labels + [CROSS_ENTROPY_IGNORE_IDX] * pad_len
            last_pos = pos[-1] if pos else 0
            pos = pos + list(range(last_pos + 1, last_pos + 1 + pad_len))

        seq_lens_padded = list(seq_lens)
        if pad_len > 0 and seq_lens_padded:
            seq_lens_padded[-1] += pad_len

        self.buf_ids = []
        self.buf_labels = []
        self.buf_pos = []
        self.buf_seq_lens = []

        return {
            "input_ids": ids,
            "labels": labels,
            "position_ids": pos,
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens_padded,
        }

    def __next__(self) -> dict:
        self._refill()
        while self.pending:
            space_left = self.pack_size - len(self.buf_ids)
            fit_idx = self._find_best_fit(space_left)

            if fit_idx is not None:
                item = self.pending.pop(fit_idx)
                ids = item["input_ids"]
                lbls = item["labels"]
                sl = len(ids)
                self.buf_ids.extend(ids)
                self.buf_labels.extend(lbls)
                self.buf_pos.extend(range(sl))
                self.buf_seq_lens.append(sl)
                self._refill()
            else:
                if self.buf_ids:
                    return self._finalize()
                self._refill()
                if not self.pending:
                    break

        if self.buf_ids:
            return self._finalize()
        raise StopIteration


class PackCPT(StreamDataset[dict]):
    """Online CPT packer with document boundary tracking.

    Concatenates tokenized text into fixed-length windows with wrap-around
    (documents split across pack boundaries). Tracks seq_lens and
    position_ids per document for block-diagonal causal attention masking,
    preventing cross-document attention contamination.

    Matches amaia's doc_causal behavior: each document gets its own
    attention block even when wrapped across pack boundaries.

    Input items: {"input_ids": list[int], "labels": list[int]}
    Output items: {"input_ids": list, "labels": list, "position_ids": list,
                   "seq_lens": list, "seq_lens_padded": list}
    """

    def __init__(
        self,
        dataset: StreamDataset[dict],
        pack_size: int,
    ) -> None:
        self.dataset = dataset
        self.pack_size = pack_size
        self.buf_ids: list[int] = []
        self.buf_labels: list[int] = []
        self.buf_pos: list[int] = []
        self.buf_seq_lens: list[int] = []
        self.leftover_ids: list[int] = []
        self.leftover_labels: list[int] = []
        self.leftover_pos_start: int = 0

    def reset(self) -> None:
        self.dataset.reset()
        self.buf_ids = []
        self.buf_labels = []
        self.buf_pos = []
        self.buf_seq_lens = []
        self.leftover_ids = []
        self.leftover_labels = []
        self.leftover_pos_start = 0

    def __next__(self) -> dict:
        while len(self.buf_ids) < self.pack_size:
            if self.leftover_ids:
                ids = self.leftover_ids
                lbls = self.leftover_labels
                pos_start = self.leftover_pos_start
                self.leftover_ids = []
                self.leftover_labels = []
                self.leftover_pos_start = 0
            else:
                try:
                    item = next(self.dataset)
                    ids = item["input_ids"]
                    lbls = item["labels"]
                    pos_start = 0
                except StopIteration:
                    break

            space = self.pack_size - len(self.buf_ids)
            if len(ids) <= space:
                chunk_len = len(ids)
                self.buf_ids.extend(ids)
                self.buf_labels.extend(lbls)
                self.buf_pos.extend(range(pos_start, pos_start + chunk_len))
                self.buf_seq_lens.append(chunk_len)
            else:
                self.buf_ids.extend(ids[:space])
                self.buf_labels.extend(lbls[:space])
                self.buf_pos.extend(range(pos_start, pos_start + space))
                self.buf_seq_lens.append(space)
                self.leftover_ids = ids[space:]
                self.leftover_labels = lbls[space:]
                self.leftover_pos_start = pos_start + space

        if len(self.buf_ids) == self.pack_size:
            result = {
                "input_ids": self.buf_ids,
                "labels": self.buf_labels,
                "position_ids": self.buf_pos,
                "seq_lens": list(self.buf_seq_lens),
                "seq_lens_padded": list(self.buf_seq_lens),
            }
            self.buf_ids = []
            self.buf_labels = []
            self.buf_pos = []
            self.buf_seq_lens = []
            return result

        raise StopIteration
