
import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone

import numpy as np
from locust import HttpUser, task
import base64
from pydub import AudioSegment
import aiohttp
import asyncio


def generate_random_audio(duration_ms, sample_rate=16000):
    """
    Generate random audio data for testing.
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz (default 16000 for Whisper)
                    Whisper models expect 16kHz audio
    """
    # Generate random data
    samples = np.random.normal(0, 1, int(sample_rate * duration_ms / 1000.0))

    # Convert to int16 array so we can make use of the pydub package
    samples = (samples * np.iinfo(np.int16).max).astype(np.int16)

    # Create an audio segment
    audio_segment = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=samples.dtype.itemsize,
        channels=1
    )

    # Convert the audio segment to a base64 string
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    base64_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_audio


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class WhisperRequestInput:
    audio_file_path: str  # Path to audio file
    api_url: str
    model: str
    model_name: Optional[str] = None
    language: Optional[str] = "en"
    temperature: float = 0.0
    response_format: str = "json"
    stream: bool = True


@dataclass
class WhisperRequestOutput:
    transcribed_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token/chunk
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    error: str = ""
    start_time: float = 0.0
    start_timestamp: str = ""


async def async_request_openai_whisper(
    request_input: WhisperRequestInput,
) -> WhisperRequestOutput:
    """
    Async function to request Whisper transcription with streaming support.
    Similar to async_request_openai_completions but for audio transcription.
    """
    api_url = request_input.api_url
    
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        output = WhisperRequestOutput()
        
        generated_text = ""
        st = time.perf_counter()
        output.start_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        output.start_time = st
        most_recent_timestamp = st
        
        try:
            # Read the audio file data first
            with open(request_input.audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Prepare the multipart form data
            data = aiohttp.FormData()
            data.add_field('model', 
                          request_input.model_name if request_input.model_name else request_input.model)
            
            if request_input.language:
                data.add_field('language', request_input.language)
            
            data.add_field('temperature', str(request_input.temperature))
            data.add_field('response_format', request_input.response_format)
            
            # Add streaming flag
            if request_input.stream:
                data.add_field('stream', 'true')
            
            # Add the audio file data
            data.add_field('file',
                          audio_data,
                          filename=os.path.basename(request_input.audio_file_path),
                          content_type='audio/wav')
            
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"
            }
            
            async with session.post(url=api_url, data=data,
                                   headers=headers) as response:
                if response.status == 200:
                    if request_input.stream:
                        # Streaming mode
                        first_chunk_received = False
                        async for line in response.content:
                            line = line.strip()
                            if not line:
                                continue
                            
                            line_str = line.decode("utf-8")
                            
                            # Handle data: prefix
                            if line_str.startswith('data: '):
                                line_str = line_str[len('data: '):]
                            
                            # Check for end of stream
                            if line_str.strip() == '[DONE]':
                                break
                            
                            try:
                                data_json = json.loads(line_str)
                                timestamp = time.perf_counter()
                                
                                # Extract content from streaming response
                                if choices := data_json.get("choices"):
                                    content = choices[0].get('delta', {}).get('content', '')
                                    
                                    # First chunk
                                    if not first_chunk_received and content:
                                        first_chunk_received = True
                                        output.ttft = timestamp - st
                                    # Subsequent chunks
                                    elif first_chunk_received:
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    
                                    generated_text += content
                                    most_recent_timestamp = timestamp
                                
                                # Check for usage info (may come at the end)
                                elif usage := data_json.get("usage"):
                                    output.output_tokens = usage.get("completion_tokens", 0)
                                    
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
                        
                        if first_chunk_received:
                            output.success = True
                        else:
                            output.success = False
                            output.error = "Never received a valid chunk to calculate TTFT."
                    else:
                        # Non-streaming mode
                        result = await response.json()
                        generated_text = result.get('text', '')
                        # Note: TTFT doesn't apply to non-streaming mode
                        # The entire response comes back at once
                        output.success = True
                    
                    output.transcribed_text = generated_text
                    output.latency = time.perf_counter() - st
                else:
                    output.error = f"HTTP {response.status}: {response.reason or ''}"
                    output.success = False
                        
        except Exception as e:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
        
        return output


class ApiUser(HttpUser):
    @task
    def send_audio_request(self):
        headers = {
            'Content-Type': 'application/json',
        }
        audio_data = generate_random_audio(1000)    # 1 second audio
        
        data = {
            "input": {
                "audio": audio_data
            }
        }
        
        self.client.post("/v2/xxxxx/runsync", json=data, headers=headers)  # Replace with your endpoint ID

async def test_whisper_function():
    """
    Test function to verify async_request_openai_whisper works correctly.
    Tests both streaming and non-streaming modes.
    """
    import tempfile
    
    print("=" * 60)
    print("Testing async_request_openai_whisper function")
    print("=" * 60)
    
    # Create a temporary audio file for testing
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as tmp_file:
        # Generate 2 seconds of random audio
        samples = np.random.normal(0, 1, int(44100 * 2))
        samples = (samples * np.iinfo(np.int16).max).astype(np.int16)
        
        audio_segment = AudioSegment(
            samples.tobytes(),
            frame_rate=44100,
            sample_width=samples.dtype.itemsize,
            channels=1
        )
        
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        tmp_file.write(buffer.getvalue())
        audio_file_path = tmp_file.name
    
    try:
        # Test 1: Streaming mode
        print("\n[Test 1] Testing STREAMING mode...")
        print("-" * 60)
        
        request_input = WhisperRequestInput(
            audio_file_path=audio_file_path,
            api_url="http://localhost:8000/v1/audio/transcriptions",
            model="openai/whisper-small",
            language="en",
            temperature=0.0,
            stream=True
        )
        
        result = await async_request_openai_whisper(request_input)
        
        print(f"Success: {result.success}")
        print(f"Transcribed Text: {result.transcribed_text[:100]}..." if len(result.transcribed_text) > 100 else f"Transcribed Text: {result.transcribed_text}")
        print(f"TTFT: {result.ttft:.4f}s")
        print(f"Latency: {result.latency:.4f}s")
        print(f"ITL count: {len(result.itl)}")
        if result.itl:
            print(f"Average ITL: {sum(result.itl)/len(result.itl):.4f}s")
        if result.error:
            print(f"Error: {result.error}")
        
        # Test 2: Non-streaming mode
        print("\n[Test 2] Testing NON-STREAMING mode...")
        print("-" * 60)
        
        request_input_no_stream = WhisperRequestInput(
            audio_file_path=audio_file_path,
            api_url="http://localhost:8000/v1/audio/transcriptions",
            model="openai/whisper-small",
            language="en",
            temperature=0.0,
            stream=False
        )
        
        result_no_stream = await async_request_openai_whisper(request_input_no_stream)
        
        print(f"Success: {result_no_stream.success}")
        print(f"Transcribed Text: {result_no_stream.transcribed_text[:100]}..." if len(result_no_stream.transcribed_text) > 100 else f"Transcribed Text: {result_no_stream.transcribed_text}")
        print(f"Latency: {result_no_stream.latency:.4f}s")
        print("(Note: TTFT doesn't apply to non-streaming - entire response arrives at once)")
        if result_no_stream.error:
            print(f"Error: {result_no_stream.error}")
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
        # Summary
        if result.success or result_no_stream.success:
            print("\n✓ At least one mode succeeded!")
            if not result.success and result_no_stream.success:
                print("  → Streaming not supported, but non-streaming works")
            elif result.success and not result_no_stream.success:
                print("  → Streaming works!")
            else:
                print("  → Both modes work!")
        else:
            print("\n✗ Both tests failed. Check the errors above.")
        
    finally:
        # Clean up temporary file
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"\nCleaned up temporary audio file: {audio_file_path}")


if __name__ == "__main__":
    import sys
    
    # Check if we want to run the test or the locust load test
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run the test function
        asyncio.run(test_whisper_function())
    else:
        # Run locust
        os.system("locust -f whisper_locustfile.py")