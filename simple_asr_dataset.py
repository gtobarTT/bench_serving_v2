"""Lightweight helper for loading small HF ASR samples.

This replaces the more complicated BenchmarkDataset/HuggingFaceDataset/ASRDataset
hierarchy when all we need is a handful of examples from a Hugging Face
dataset such as ``hf-internal-testing/librispeech_asr_dummy``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from datasets import Audio, load_dataset


@dataclass
class SimpleASRDataset:
    """Minimal dataset loader for Hugging Face ASR datasets."""

    dataset_path: str = "hf-internal-testing/librispeech_asr_dummy"
    dataset_subset: Optional[str] = "clean"
    dataset_split: str = "validation"
    seed: Optional[int] = 42

    def sample(self, sample_size: int = 10) -> List[dict]:
        """Return ``sample_size`` rows from the dataset as plain dicts."""

        dataset = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=False,
        )

        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(decode=False))

        if self.seed is not None:
            dataset = dataset.shuffle(seed=self.seed)

        available = len(dataset)
        if available == 0:
            return []

        requested = max(sample_size, 1)
        if requested <= available:
            selected = dataset.select(range(requested))
        else:
            repeats = math.ceil(requested / available)
            indices = (list(range(available)) * repeats)[:requested]
            selected = dataset.select(indices)

        return [selected[i] for i in range(len(selected))]


def load_librispeech_dummy_sample(sample_size: int = 10) -> List[dict]:
    """Convenience wrapper that returns a small sample from Librispeech dummy."""

    return SimpleASRDataset().sample(sample_size=sample_size)


