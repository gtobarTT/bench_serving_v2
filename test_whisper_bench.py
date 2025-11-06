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
import numpy as np
import soundfile as sf

# Add the current directory to path to import from whisper_locustfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_locustfile import async_request_openai_whisper, WhisperRequestInput
from simple_asr_dataset import LocalASRDataset


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


def calculate_tokens_per_second_metrics(results):
    """Calculate tokens per second metrics similar to benchmark_serving.py."""
    outputs = [res for _, res in results]
    
    successful_indices = [i for i, o in enumerate(outputs) if o.success]
    if not successful_indices:
        return 0.0, 0.0, 0
    
    min_start_time = min(outputs[i].start_time for i in successful_indices)
    max_end_time = max(outputs[i].start_time + outputs[i].latency
                       for i in successful_indices)
    duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
    tokens_per_second = np.zeros(duration_seconds)
    concurrent_requests_per_second = np.zeros(duration_seconds)
    
    for i in successful_indices:
        output = outputs[i]
        st = output.start_time
        
        # Token emission timestamps
        token_times = [st + output.ttft]
        first_token_time = st + output.ttft
        current_time = token_times[0]
        for itl_value in output.itl:
            current_time += itl_value
            if current_time > first_token_time:  # redundant check
                token_times.append(current_time)
        
        for token_time in token_times:
            second_bucket = int(token_time - min_start_time)
            if 0 <= second_bucket < duration_seconds:
                tokens_per_second[second_bucket] += 1
        
        request_start_second = int(st - min_start_time)
        request_end_second = int((st + output.latency) - min_start_time)
        for second in range(request_start_second, request_end_second + 1):
            if 0 <= second < duration_seconds:
                concurrent_requests_per_second[second] += 1
    
    max_tokens_per_s = float(np.max(tokens_per_second)) if len(tokens_per_second) > 0 else 0.0
    avg_tokens_per_s = float(np.mean(tokens_per_second[tokens_per_second > 0])) if np.any(tokens_per_second > 0) else 0.0
    max_concurrent = int(np.max(concurrent_requests_per_second)) if len(concurrent_requests_per_second) > 0 else 0
    
    return max_tokens_per_s, avg_tokens_per_s, max_concurrent


async def run_concurrent_batch(audio_file_paths: List[str], session: aiohttp.ClientSession, max_concurrent: int = 32):
    """Run requests with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_request(idx: int, path: str):
        async with semaphore:
            return await run_streaming_request(idx, path, session)
    
    tasks = [bounded_request(idx, path) for idx, path in enumerate(audio_file_paths)]
    return await asyncio.gather(*tasks)


async def main():
    print("=" * 70)
    print("Testing async_request_openai_whisper Function")
    print("=" * 70)

    print("\n[Setup] Loading sample dataset rows...")
    dataset = LocalASRDataset()
    samples = dataset.sample()
    print(f"Loaded {len(samples)} sample rows from local dataset ({dataset.data_dir}).")

    if not samples:
        print("Dataset returned no samples. Exiting.")
        return

    # Configuration
    num_sequential = 10  # Number of samples for sequential testing
    max_concurrent = 32  # Maximum concurrent requests for batch testing
    
    audio_file_paths: List[str] = []
    for idx, sample in enumerate(samples):
        audio_array, sampling_rate = extract_audio_array_and_rate(sample)
        duration_seconds = len(audio_array) / float(sampling_rate) if sampling_rate else 0.0
        # print(f"Sample {idx}: {len(audio_array)} samples @ {sampling_rate} Hz (~{duration_seconds:.2f}s)")
        path = create_audio_file_from_dataset_sample(sample)
        audio_file_paths.append(path)
        # print(f"Saved sample {idx} to temporary WAV: {path}")

    try:
        async with aiohttp.ClientSession() as session:
            # Warm-up phase: Send a few requests to warm up the server
            print("\n" + "=" * 70)
            print("[Warm-up] Sending warm-up requests to prepare server...")
            print("=" * 70)
            warmup_count = min(3, len(audio_file_paths))
            for i in range(warmup_count):
                print(f"Warm-up request {i+1}/{warmup_count}...")
                await run_streaming_request(i, audio_file_paths[i], session)
            print("Warm-up complete!")
            
            # Sequential testing: First 10 samples
            print("\n" + "=" * 70)
            print(f"[Sequential] Testing first {num_sequential} samples")
            print("=" * 70)
            sequential_results = []
            for idx, path in enumerate(audio_file_paths[:num_sequential]):
                print("\n" + "-" * 70)
                print(f"[Sequential] Streaming request for sample {idx}")
                print("-" * 70)
                idx_result = await run_streaming_request(idx, path, session)
                sequential_results.append(idx_result)
                _, result = idx_result
                # if result.success:
                #     print(f"  TTFT: {result.ttft*1000:.2f} ms | Latency: {result.latency*1000:.2f} ms | Tokens: {result.output_tokens}")
                # else:
                #     print(f"  Error: {result.error}")

            # Batch testing: All samples with max 32 concurrent
            print("\n" + "=" * 70)
            print(f"[Batch] Testing all {len(audio_file_paths)} samples with max {max_concurrent} concurrent requests")
            print("=" * 70)
            batch_results = await run_concurrent_batch(audio_file_paths, session, max_concurrent=max_concurrent)

        def summarize(results, label: str):
            print("\n" + "-" * 70)
            print(f"Summary: {label}")
            print("-" * 70)
            success_count = 0
            failed_count = 0
            total_ttft = 0.0
            total_latency = 0.0
            
            for idx, res in results:
                if res.success:
                    success_count += 1
                    total_ttft += res.ttft
                    total_latency += res.latency
                    print(f"Sample {idx}: TTFT {res.ttft*1000:.2f} ms, Latency {res.latency*1000:.2f} ms, Tokens {res.output_tokens}")
                else:
                    failed_count += 1
                    print(f"Sample {idx}: FAILED -> {res.error}")
            
            print("-" * 70)
            print(f"Success: {success_count}/{len(results)}, Failed: {failed_count}/{len(results)}")
            avg_ttft = (total_ttft/success_count)*1000 if success_count > 0 else 0
            avg_latency = (total_latency/success_count)*1000 if success_count > 0 else 0
            
            # Calculate tokens per second metrics
            max_tokens_per_s, avg_tokens_per_s, max_concurrent = calculate_tokens_per_second_metrics(results)
            
            if success_count > 0:
                print(f"Average TTFT: {avg_ttft:.2f} ms")
                print(f"Average Latency: {avg_latency:.2f} ms")
                print(f"Average Tokens/s: {avg_tokens_per_s:.2f}")
                print(f"Max Tokens/s: {max_tokens_per_s:.2f}")
                print(f"Max Concurrent Requests: {max_concurrent}")
            print("-" * 70)
            return avg_ttft, avg_latency, avg_tokens_per_s, success_count, failed_count

        seq_ttft, seq_latency, seq_tokens_per_s, seq_success, seq_failed = summarize(sequential_results, f"Sequential ({num_sequential} samples)")
        batch_ttft, batch_latency, batch_tokens_per_s, batch_success, batch_failed = summarize(batch_results, f"Batch ({len(audio_file_paths)} samples, max {max_concurrent} concurrent)")
        
        # Final comparison summary
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Sequential Mode:")
        print(f"  Average TTFT:      {seq_ttft:.2f} ms")
        print(f"  Average Latency:   {seq_latency:.2f} ms")
        print(f"  Average Tokens/s:  {seq_tokens_per_s:.2f}")
        print(f"  Success Rate:      {seq_success}/{seq_success + seq_failed}")
        print()
        print(f"Batch Mode (max {max_concurrent} concurrent):")
        print(f"  Average TTFT:      {batch_ttft:.2f} ms")
        print(f"  Average Latency:   {batch_latency:.2f} ms")
        print(f"  Average Tokens/s:  {batch_tokens_per_s:.2f}")
        print(f"  Success Rate:      {batch_success}/{batch_success + batch_failed}")
        print()
        if seq_ttft > 0 and batch_ttft > 0:
            ttft_diff = batch_ttft - seq_ttft
            ttft_pct = (ttft_diff / seq_ttft) * 100
            print(f"TTFT Difference:      {ttft_diff:+.2f} ms ({ttft_pct:+.1f}%)")
        if seq_tokens_per_s > 0 and batch_tokens_per_s > 0:
            tokens_diff = batch_tokens_per_s - seq_tokens_per_s
            tokens_pct = (tokens_diff / seq_tokens_per_s) * 100
            print(f"Tokens/s Difference:  {tokens_diff:+.2f} tokens/s ({tokens_pct:+.1f}%)")
        print("=" * 70)

    finally:
        print("\n[Cleanup] Removing temporary audio files...")
        for path in audio_file_paths:
            if os.path.exists(path):
                os.remove(path)
        print(f"[Cleanup] Removed {len(audio_file_paths)} temporary audio files")


if __name__ == "__main__":
    asyncio.run(main())


