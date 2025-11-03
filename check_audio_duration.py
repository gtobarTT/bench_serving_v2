#!/usr/bin/env python3
"""
Quick script to check the duration of generated audio.
"""
import wave
import tempfile
import base64
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whisper_locustfile import generate_random_audio


def check_audio_duration_from_base64(base64_audio):
    """Check duration of base64-encoded WAV audio."""
    # Decode base64 to bytes
    audio_bytes = base64.b64decode(base64_audio)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Read WAV file properties
        with wave.open(tmp_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            duration = frames / float(rate)
            
            print(f"Audio Properties:")
            print(f"  Sample Rate: {rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample Width: {sample_width} bytes")
            print(f"  Total Frames: {frames:,}")
            print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            return duration
    finally:
        # Clean up
        os.remove(tmp_path)


def check_audio_duration_from_file(file_path):
    """Check duration of a WAV file."""
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        duration = frames / float(rate)
        
        print(f"Audio Properties for: {file_path}")
        print(f"  Sample Rate: {rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Sample Width: {sample_width} bytes")
        print(f"  Total Frames: {frames:,}")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        return duration


if __name__ == "__main__":
    print("=" * 70)
    print("Audio Duration Checker")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Check a file provided as argument
        file_path = sys.argv[1]
        print(f"\nChecking file: {file_path}\n")
        check_audio_duration_from_file(file_path)
    else:
        # Generate and check audio at different durations
        test_durations_ms = [1000, 5000, 10000, 30000, 60000]
        
        print("\nGenerating and checking audio at various durations...\n")
        
        for duration_ms in test_durations_ms:
            print(f"\n{'-' * 70}")
            print(f"Testing: {duration_ms}ms ({duration_ms/1000:.1f}s) of audio")
            print(f"{'-' * 70}")
            
            # Generate audio
            print("Generating audio...")
            base64_audio = generate_random_audio(duration_ms)
            
            # Check actual duration
            actual_duration = check_audio_duration_from_base64(base64_audio)
            expected_duration = duration_ms / 1000.0
            
            # Compare
            diff = abs(actual_duration - expected_duration)
            if diff < 0.01:  # Within 10ms
                print(f"\n✓ PASS: Expected {expected_duration:.2f}s, got {actual_duration:.2f}s")
            else:
                print(f"\n✗ FAIL: Expected {expected_duration:.2f}s, got {actual_duration:.2f}s (diff: {diff:.3f}s)")
        
        print("\n" + "=" * 70)
        print("All checks complete!")
        print("=" * 70)

