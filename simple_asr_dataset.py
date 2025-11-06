"""Lightweight helper for loading small HF ASR samples.

This replaces the more complicated BenchmarkDataset/HuggingFaceDataset/ASRDataset
hierarchy when all we need is a handful of examples from a Hugging Face
dataset such as ``hf-internal-testing/librispeech_asr_dummy``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from datasets import Audio, load_dataset


@dataclass
class SimpleASRDataset:
    """Minimal dataset loader for Hugging Face ASR datasets."""

    dataset_path: str = "openslr/librispeech_asr"
    dataset_subset: Optional[str] = "clean"
    dataset_split: str = "test"
    seed: Optional[int] = 42

    def sample(self, sample_size: Optional[int | str] = None) -> List[dict]:
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

        if sample_size is None:
            return dataset
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


@dataclass
class LocalASRDataset:
    """Loader for locally saved ASR samples (e.g., preprocessed 30s clips)."""

    data_dir: str = "/home/ubuntu/bench_serving_v2/librispeech_30s_samples"
    seed: Optional[int] = 42

    def sample(self, sample_size: Optional[int] = None) -> List[dict]:
        """Return ``sample_size`` rows from the local dataset as plain dicts."""
        
        data_path = Path(self.data_dir)
        metadata_path = data_path / "metadata.json"
        audio_dir = data_path / "audio"
        
        # Load metadata
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Please run the notebook to generate 30s samples first."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        samples_metadata = metadata.get('samples', [])
        
        if not samples_metadata:
            return []
        
        # Shuffle if seed is provided
        if self.seed is not None:
            import random
            rng = random.Random(self.seed)
            samples_metadata = samples_metadata.copy()
            rng.shuffle(samples_metadata)
        
        # Determine how many samples to return
        available = len(samples_metadata)
        if sample_size is None:
            requested = available
        else:
            requested = min(sample_size, available)
        
        # Load the requested samples
        result = []
        for sample_meta in samples_metadata[:requested]:
            audio_path = audio_dir / sample_meta['filename']
            
            # Read audio bytes
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Create dict in compatible format with SimpleASRDataset
            result.append({
                'id': f"{sample_meta['speaker_id']}-{sample_meta['chapter_id']}-{sample_meta['segment_index']:04d}",
                'audio': {
                    'bytes': audio_bytes,
                    'path': sample_meta['filename'],
                },
                'text': sample_meta['text'],
                'speaker_id': sample_meta['speaker_id'],
                'chapter_id': sample_meta['chapter_id'],
                'duration': sample_meta['duration'],
                'sample_rate': sample_meta['sample_rate'],
            })
        
        return result


if __name__ == "__main__":
    dataset = load_librispeech_dummy_sample(sample_size=10)
    print(dataset)
    
