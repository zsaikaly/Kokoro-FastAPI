import os
import time
import json
import scipy.io.wavfile as wavfile
import tiktoken
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(enc.encode(text))

def get_audio_length(filepath: str) -> float:
    """Get audio length in seconds"""
    # Convert API path to local path
    local_path = filepath.replace('/app/api/src/output', 'api/src/output')
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Audio file not found at {local_path} (from {filepath})")
    rate, data = wavfile.read(local_path)
    return len(data) / rate

def make_tts_request(text: str, timeout: int = 120) -> tuple[float, float]:
    """Make TTS request and return processing time and output length"""
    try:
        # Submit request
        response = requests.post(
            'http://localhost:8880/tts',
            json={'text': text},
            timeout=timeout
        )
        request_id = response.json()['request_id']
        
        # Poll until complete
        start_time = time.time()
        while True:
            status_response = requests.get(
                f'http://localhost:8880/tts/{request_id}',
                timeout=timeout
            )
            status = status_response.json()
            
            if status['status'] == 'completed':
                # Convert Docker path to local path
                docker_path = status['output_file']
                filename = os.path.basename(docker_path)  # Get just the filename
                local_path = os.path.join('api/src/output', filename)  # Construct local path
                try:
                    audio_length = get_audio_length(local_path)
                    return status['processing_time'], audio_length
                except Exception as e:
                    print(f"Error reading audio file: {str(e)}")
                    return None, None
            
            if time.time() - start_time > timeout:
                raise TimeoutError()
                
            time.sleep(0.5)
            
    except (requests.exceptions.Timeout, TimeoutError):
        print(f"Request timed out for text: {text[:50]}...")
        return None, None
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return None, None

def main():
    # Create output directory
    os.makedirs('examples/output', exist_ok=True)
    
    # Read input text
    with open('examples/the_time_machine_hg_wells.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create range of sizes up to full text
    sizes = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, len(text)]
    
    # Process chunks
    results = []
    for size in sizes:
        # Get chunk and count tokens
        chunk = text[:size]
        num_tokens = count_tokens(chunk)
        
        print(f"\nProcessing chunk with {num_tokens} tokens ({size} chars):")
        print(f"Text preview: {chunk[:100]}...")
        
        processing_time, audio_length = make_tts_request(chunk)
        
        if processing_time is not None:
            results.append({
                'char_length': size,
                'tokens': num_tokens,
                'processing_time': processing_time,
                'output_length': audio_length
            })
    with open('examples/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Plot 1: Processing Time vs Output Length
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='output_length', y='processing_time')
    sns.regplot(data=df, x='output_length', y='processing_time', scatter=False)
    plt.title('Processing Time vs Output Length')
    plt.xlabel('Output Audio Length (seconds)')
    plt.ylabel('Processing Time (seconds)')
    plt.savefig('examples/time_vs_output.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Processing Time vs Token Count
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='tokens', y='processing_time')
    sns.regplot(data=df, x='tokens', y='processing_time', scatter=False)
    plt.title('Processing Time vs Token Count')
    plt.xlabel('Number of Input Tokens')
    plt.ylabel('Processing Time (seconds)')
    plt.savefig('examples/time_vs_tokens.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to examples/benchmark_results.json")
    print("Plots saved as time_vs_output.png and time_vs_tokens.png")

if __name__ == '__main__':
    main()
