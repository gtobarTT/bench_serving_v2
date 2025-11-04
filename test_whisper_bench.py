#!/usr/bin/env python3
"""
Simple test script to verify the async_request_openai_whisper function works.
"""
import asyncio
import io
import os
import sys
import tempfile
import base64
import wave
from typing import List

import aiohttp
import soundfile as sf

# Add the current directory to path to import from whisper_locustfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_locustfile import async_request_openai_whisper, WhisperRequestInput, generate_random_audio
from simple_asr_dataset import SimpleASRDataset


def create_test_audio_file():
    base64_audio = generate_random_audio(120000)
    audio_bytes = base64.b64decode(base64_audio)
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        return tmp_file.name


def extract_audio_array_and_rate(sample: dict):
    audio_meta = sample.get("audio", {})
    audio_array = audio_meta.get("array")
    sampling_rate = audio_meta.get("sampling_rate")
    audio_path = audio_meta.get("path")
    audio_bytes = audio_meta.get("bytes")

    if audio_array is None:
        if audio_bytes is not None:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as snd_file:
                audio_array = snd_file.read(dtype="float32")
                sampling_rate = snd_file.samplerate
        elif audio_path:
            audio_array, sampling_rate = sf.read(audio_path)
        else:
            raise ValueError(
                "Dataset sample must include audio bytes, array, or a file path."
            )
    elif sampling_rate is None:
        if audio_bytes is not None:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as snd_file:
                sampling_rate = snd_file.samplerate
        elif audio_path:
            _audio_array, sampling_rate = sf.read(audio_path)
        else:
            raise ValueError("Audio sample missing sampling rate and path.")

    return audio_array, sampling_rate


def create_audio_file_from_dataset_sample(sample: dict) -> str:
    audio_array, sampling_rate = extract_audio_array_and_rate(sample)

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_array, sampling_rate, format="WAV")
        return tmp_file.name


async def run_streaming_request(idx: int, audio_file_path: str, session: aiohttp.ClientSession):
    request_input = WhisperRequestInput(
        audio_file_path=audio_file_path,
        api_url="http://localhost:8000/v1/audio/transcriptions",
        model="openai/whisper-large-v3",
        language="en",
        temperature=0.0,
        stream=True,
    )
    result = await async_request_openai_whisper(request_input, session=session)
    return idx, result


async def main():
    print("=" * 70)
    print("Testing async_request_openai_whisper Function")
    print("=" * 70)

    print("\n[Setup] Loading sample dataset rows...")
    dataset = SimpleASRDataset()
    samples = dataset.sample(sample_size=10)
    print(f"Loaded {len(samples)} sample rows from {dataset.dataset_path} ({dataset.dataset_split}).")

    if not samples:
        print("Dataset returned no samples. Exiting.")
        return

    audio_file_paths: List[str] = []
    for idx, sample in enumerate(samples):
        audio_array, sampling_rate = extract_audio_array_and_rate(sample)
        duration_seconds = len(audio_array) / float(sampling_rate) if sampling_rate else 0.0
        print(f"Sample {idx}: {len(audio_array)} samples @ {sampling_rate} Hz (~{duration_seconds:.2f}s)")
        path = create_audio_file_from_dataset_sample(sample)
        audio_file_paths.append(path)
        print(f"Saved sample {idx} to temporary WAV: {path}")

    try:
        async with aiohttp.ClientSession() as session:
            sequential_results = []
            for idx, path in enumerate(audio_file_paths):
                print("\n" + "=" * 70)
                print(f"[Sequential] Streaming request for sample {idx}")
                print("=" * 70)
                idx_result = await run_streaming_request(idx, path, session)
                sequential_results.append(idx_result)
                _, result = idx_result
                if result.success:
                    print(f"  TTFT: {result.ttft*1000:.2f} ms | Latency: {result.latency*1000:.2f} ms | Tokens: {result.output_tokens}")
                else:
                    print(f"  Error: {result.error}")

            print("\n" + "=" * 70)
            print("[Batch] Launching streaming requests concurrently")
            print("=" * 70)
            batch_results = await asyncio.gather(
                *(run_streaming_request(idx, path, session) for idx, path in enumerate(audio_file_paths))
            )

        def summarize(results, label: str):
            print("\n" + "-" * 70)
            print(f"Summary: {label}")
            for idx, res in results:
                if res.success:
                    print(f"Sample {idx}: TTFT {res.ttft*1000:.2f} ms, Latency {res.latency*1000:.2f} ms, Tokens {res.output_tokens}")
                else:
                    print(f"Sample {idx}: FAILED -> {res.error}")

        summarize(sequential_results, "Sequential")
        summarize(batch_results, "Batch")

    finally:
        for path in audio_file_paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"[Cleanup] Removed temporary audio file: {path}")


if __name__ == "__main__":
    asyncio.run(main())


