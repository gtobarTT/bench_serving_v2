## Whisper Benchmarking Function Reference

### Overview
The Whisper benchmarking flow in this workspace combines helper utilities for dataset preparation, temporary audio file generation, and request execution against the transcription endpoint. The core components live in `whisper_locustfile.py`, `simple_asr_dataset.py`, and `test_whisper_bench.py`. This report documents the key functions, their responsibilities, and how they interact during benchmarking.

### `whisper_locustfile.py`

- **`generate_random_audio(duration_ms, sample_rate=16000)`** – Produces synthetic single-channel WAV audio by sampling Gaussian noise, exporting it via `pydub`, and returning base64-encoded bytes. Useful for fallback testing when no dataset audio is available.

- **`StreamedResponseHandler.add_chunk(chunk_bytes)`** – Buffers streaming Server-Sent Event (SSE) chunks, splits them on double newlines, and returns complete `data:` messages while handling partial JSON payloads and the `[DONE]` terminator.

- **`async_request_openai_whisper(request_input, session=None)`** – Core async client used by the benchmarks. It constructs multipart form data (model, language, streaming flags, audio file) and posts to `/v1/audio/transcriptions`. Streaming responses are parsed chunk-by-chunk to measure time-to-first-token (TTFT), inter-token latencies (ITL), collect generated text, and capture usage metrics or error bodies.

```80:228:/home/ubuntu/bench_serving_v2/whisper_locustfile.py
# ... existing code ...
async def async_request_openai_whisper(
    request_input: WhisperRequestInput,
    session: Optional[aiohttp.ClientSession] = None,
) -> WhisperRequestOutput:
    # ... existing code ...
```

### `simple_asr_dataset.py`

- **`SimpleASRDataset.sample(sample_size=10)`** – Thin wrapper around `datasets.load_dataset` that shuffles (optional seed), casts the `audio` column with `Audio(decode=False)` to avoid torch dependencies, repeats rows if sample size exceeds dataset length, and returns plain Python dicts ready for benchmarking.

- **`load_librispeech_dummy_sample(sample_size=10)`** – Convenience function that instantiates `SimpleASRDataset` with default Librispeech dummy parameters and delegates to `sample`.

```26:58:/home/ubuntu/bench_serving_v2/simple_asr_dataset.py
def sample(self, sample_size: int = 10) -> List[dict]:
    # ... existing code ...

def load_librispeech_dummy_sample(sample_size: int = 10) -> List[dict]:
    # ... existing code ...
```

### `test_whisper_bench.py`

- **`create_test_audio_file()`** – Generates ≈60 seconds of random audio using `generate_random_audio`, decodes from base64, and writes the result to a temporary WAV for legacy/manual testing scenarios.

- **`extract_audio_array_and_rate(sample)`** – Normalizes Hugging Face audio samples by extracting waveform and sampling rate from pre-decoded arrays, raw bytes, or file paths (`sf.read`) so all downstream steps operate on NumPy arrays without requiring torch.

- **`create_audio_file_from_dataset_sample(sample)`** – Calls `extract_audio_array_and_rate`, then persists the waveform to a temporary WAV on disk for benchmarking.

- **`run_streaming_request(idx, audio_file_path, session)`** – Wraps `async_request_openai_whisper` to execute a single streaming transcription with a shared `aiohttp.ClientSession`, returning both the sample index and the structured result object.

- **`main()`** – Orchestrates the end-to-end benchmark:
  1. Loads 10 Librispeech dummy samples via `SimpleASRDataset`.
  2. Converts each sample to a temporary WAV and records basic metadata (sample count, sampling rate, duration).
  3. Runs sequential streaming requests to gather per-sample TTFT/latency/token stats.
  4. Launches concurrent (batch) streaming requests with `asyncio.gather` to measure parallel performance using the same HTTP session.
  5. Summarizes sequential vs. batch metrics and cleans up all temporary files.

```23:130:/home/ubuntu/bench_serving_v2/test_whisper_bench.py
def create_test_audio_file():
    # ... existing code ...

def extract_audio_array_and_rate(sample: dict):
    # ... existing code ...

async def run_streaming_request(idx: int, audio_file_path: str, session: aiohttp.ClientSession):
    # ... existing code ...

async def main():
    # ... existing code ...
```

### Execution Flow Summary
1. **Dataset Preparation** – `SimpleASRDataset.sample` (or fallback `create_test_audio_file`) provides audio payloads and metadata.
2. **Temporary WAV Creation** – `extract_audio_array_and_rate` → `create_audio_file_from_dataset_sample` ensures each request has a local WAV file to send.
3. **Benchmark Runs** – `main` drives sequential and batch invocations of `run_streaming_request`, which in turn relies on `async_request_openai_whisper`.
4. **Streaming Metrics** – `StreamedResponseHandler` and the logic inside `async_request_openai_whisper` track TTFT, ITL, latency, token counts, and errors for reporting.

These functions collectively enable reproducible Whisper transcription benchmarks against the `/v1/audio/transcriptions` endpoint using both synthetic and dataset-backed audio samples.

