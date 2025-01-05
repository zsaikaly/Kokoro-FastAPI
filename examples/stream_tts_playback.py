#!/usr/bin/env python3
import requests
import sounddevice as sd
import numpy as np
import time
import os
import wave

def play_streaming_tts(text: str, output_file: str = None, voice: str = "af"):
    """Stream TTS audio and play it back in real-time"""
    
    print("\nStarting TTS stream request...")
    start_time = time.time()
    
    # Initialize variables
    sample_rate = 24000  # Known sample rate for Kokoro
    audio_started = False
    stream = None
    chunk_count = 0
    total_bytes = 0
    first_chunk_time = None
    all_audio_data = bytearray()  # Raw PCM audio data
    
    # Make streaming request to API
    try:
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": "pcm",
                "stream": True
            },
            stream=True,
            timeout=1800
        )
        response.raise_for_status()
        print(f"Request started successfully after {time.time() - start_time:.2f}s")
        
        # Process streaming response
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                chunk_count += 1
                total_bytes += len(chunk)
                
                # Handle first chunk
                if not audio_started:
                    first_chunk_time = time.time()
                    print(f"\nReceived first chunk after {first_chunk_time - start_time:.2f}s")
                    print(f"First chunk size: {len(chunk)} bytes")
                    
                    # Accumulate raw audio data
                    all_audio_data.extend(chunk)
                    
                    # Convert PCM to float32 for playback
                    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    # Scale to [-1, 1] range for sounddevice
                    audio_data = audio_data / 32768.0
                    
                    # Start audio stream
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype=np.float32
                    )
                    stream.start()
                    audio_started = True
                    print("Audio playback started")
                    
                    # Play first chunk
                    if len(audio_data) > 0:
                        stream.write(audio_data)
                    
                # Handle subsequent chunks
                else:
                    # Accumulate raw audio data
                    all_audio_data.extend(chunk)
                    
                    # Convert PCM to float32 for playback
                    audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                    audio_data = audio_data / 32768.0
                    if len(audio_data) > 0:
                        stream.write(audio_data)
                        
                # Log progress every 10 chunks
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Progress: {chunk_count} chunks, {total_bytes/1024:.1f}KB received, {elapsed:.1f}s elapsed")
        
        # Final stats
        total_time = time.time() - start_time
        print(f"\nStream complete:")
        print(f"Total chunks: {chunk_count}")
        print(f"Total data: {total_bytes/1024:.1f}KB")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average speed: {(total_bytes/1024)/total_time:.1f}KB/s")
        
        # Save as WAV file
        if output_file:
            print(f"\nWriting audio to {output_file}")
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(all_audio_data)
            print(f"Saved {len(all_audio_data)} bytes of audio data")
        
        # Clean up
        if stream is not None:
            stream.stop()
            stream.close()
            
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error - Is the server running? Error: {str(e)}")
        if stream is not None:
            stream.stop()
            stream.close()
    except Exception as e:
        print(f"Error during streaming: {str(e)}")
        if stream is not None:
            stream.stop()
            stream.close()

def main():
    # Load sample text from HG Wells
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wells_path = os.path.join(script_dir, "assorted_checks/benchmarks/the_time_machine_hg_wells.txt")
    output_path = os.path.join(script_dir, "output.wav")
    
    with open(wells_path, "r", encoding="utf-8") as f:
        full_text = f.read()
        # Take first few paragraphs
        text = " ".join(full_text.split("\n\n")[:2])
    
    print("\nStarting TTS stream playback...")
    print(f"Text length: {len(text)} characters")
    print("\nFirst 100 characters:")
    print(text[:100] + "...")
    
    play_streaming_tts(text, output_file=output_path)

if __name__ == "__main__":
    main()
