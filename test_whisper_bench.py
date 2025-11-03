#!/usr/bin/env python3
"""
Simple test script to verify the async_request_openai_whisper function works.
"""
import asyncio
import os
import sys
import tempfile
import base64
import wave

# Add the current directory to path to import from whisper_locustfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_locustfile import async_request_openai_whisper, WhisperRequestInput, generate_random_audio


def create_test_audio_file():
    """Create a simple test audio file using generate_random_audio from whisper_locustfile."""
    # Generate 60 seconds of random audio using the existing function
    base64_audio = generate_random_audio(120000)  # 60000ms = 60 seconds
    
    # Decode the base64 audio and save to a temporary file
    audio_bytes = base64.b64decode(base64_audio)
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name
    
    # Verify the audio duration
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        print(f"Audio file created: {duration:.2f}s ({rate}Hz, {wav_file.getnchannels()} channel(s))")
    
    return audio_path


async def main():
    print("=" * 70)
    print("Testing async_request_openai_whisper Function")
    print("=" * 70)
    
    # Create test audio file
    print("\n[Setup] Creating test audio file...")
    audio_file_path = create_test_audio_file()
    print(f"Created audio file: {audio_file_path}")
    
    try:
        # Test 1: Streaming mode
        print("\n" + "=" * 70)
        print("[TEST 1] Streaming Mode")
        print("=" * 70)
        
        request_input = WhisperRequestInput(
            audio_file_path=audio_file_path,
            api_url="http://localhost:8000/v1/audio/transcriptions",
            model="openai/whisper-large-v3",
            language="en",
            temperature=0.0,
            stream=True
        )
        
        print(f"API URL: {request_input.api_url}")
        print(f"Model: {request_input.model}")
        print(f"Language: {request_input.language}")
        print(f"Stream: {request_input.stream}")
        print("\nSending request...")
        
        result = await async_request_openai_whisper(request_input)
        
        print(f"\n✓ Success: {result.success}")
        if result.success:
            print(f"  Transcribed Text: '{result.transcribed_text[:100]}...'" if len(result.transcribed_text) > 100 else f"  Transcribed Text: '{result.transcribed_text}'")
            print(f"  Time to First Token (TTFT): {result.ttft*1000:.4f}ms")
            print(f"  Total Latency: {result.latency*1000:.4f}ms")
            print(f"  Inter-token Latencies: {len(result.itl)} chunks")
            if result.itl:
                print(f"  Average ITL: {sum(result.itl)/len(result.itl)*1000:.4f}ms")
                print(f"  Min ITL: {min(result.itl)*1000:.4f}ms")
                print(f"  Max ITL: {max(result.itl)*1000:.4f}ms")
        else:
            print(f"  Error: {result.error}")
        
        # Test 2: Non-streaming mode
        print("\n" + "=" * 70)
        print("[TEST 2] Non-Streaming Mode")
        print("=" * 70)
        
        request_input_no_stream = WhisperRequestInput(
            audio_file_path=audio_file_path,
            api_url="http://localhost:8000/v1/audio/transcriptions",
            model="openai/whisper-large-v3",
            language="en",
            temperature=0.0,
            stream=False
        )
        
        print(f"API URL: {request_input_no_stream.api_url}")
        print(f"Model: {request_input_no_stream.model}")
        print(f"Stream: {request_input_no_stream.stream}")
        print("\nSending request...")
        
        result_no_stream = await async_request_openai_whisper(request_input_no_stream)
        
        print(f"\n✓ Success: {result_no_stream.success}")
        if result_no_stream.success:
            print(f"  Transcribed Text: '{result_no_stream.transcribed_text[:100]}...'" if len(result_no_stream.transcribed_text) > 100 else f"  Transcribed Text: '{result_no_stream.transcribed_text}'")
            print(f"  Total Latency: {result_no_stream.latency*1000:.4f}ms")
            print(f"  (Note: TTFT doesn't apply to non-streaming - entire response arrives at once)")
        else:
            print(f"  Error: {result_no_stream.error}")
        
        # Summary
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        
        if result.success or result_no_stream.success:
            print("\n✓ SUCCESS: At least one mode works!")
            if result.success and result_no_stream.success:
                print("  → Both streaming and non-streaming modes work!")
            elif result.success:
                print("  → Streaming mode works!")
                print("  → Non-streaming mode failed (this is OK if not supported)")
            else:
                print("  → Non-streaming mode works!")
                print("  → Streaming mode not supported (falling back to non-streaming)")
        else:
            print("\n✗ FAILURE: Both tests failed.")
            print("\nPlease check:")
            print("  1. Is the vLLM server running on http://localhost:8000?")
            print("  2. Is the Whisper model loaded?")
            print("  3. Check the error messages above for details.")
        
    finally:
        # Clean up
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"\n[Cleanup] Removed temporary audio file: {audio_file_path}")


if __name__ == "__main__":
    asyncio.run(main())

