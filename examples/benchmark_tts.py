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
    sizes = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
    
    # Process chunks
    results = []
    import random
    for size in sizes:
        # Get random starting point ensuring we have enough text left
        max_start = len(text) - size
        if max_start > 0:
            start = random.randint(0, max_start)
            chunk = text[start:start + size]
        else:
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
    
    # Set the style
    sns.set_theme(style="darkgrid", palette="husl", font_scale=1.1)
    
    # Common plot settings
    def setup_plot(fig, ax, title):
        # Improve grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set title and labels with better fonts
        ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
        ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='medium')
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='medium')
        
        # Improve tick labels
        ax.tick_params(labelsize=10)
        
        # Add subtle spines
        for spine in ax.spines.values():
            spine.set_color('#666666')
            spine.set_linewidth(0.5)
            
        return fig, ax
    
    # Plot 1: Processing Time vs Output Length
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with custom styling
    scatter = sns.scatterplot(data=df, x='output_length', y='processing_time', 
                            s=100, alpha=0.6, color='#2ecc71')
    
    # Add regression line with confidence interval
    sns.regplot(data=df, x='output_length', y='processing_time', 
                scatter=False, color='#e74c3c', line_kws={'linewidth': 2})
    
    # Calculate correlation
    corr = df['output_length'].corr(df['processing_time'])
    
    # Add correlation annotation
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=ax.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    setup_plot(fig, ax, 'Processing Time vs Output Length')
    ax.set_xlabel('Output Audio Length (seconds)')
    ax.set_ylabel('Processing Time (seconds)')
    
    plt.savefig('examples/time_vs_output.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Processing Time vs Token Count
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with custom styling
    scatter = sns.scatterplot(data=df, x='tokens', y='processing_time', 
                            s=100, alpha=0.6, color='#3498db')
    
    # Add regression line with confidence interval
    sns.regplot(data=df, x='tokens', y='processing_time', 
                scatter=False, color='#e74c3c', line_kws={'linewidth': 2})
    
    # Calculate correlation
    corr = df['tokens'].corr(df['processing_time'])
    
    # Add correlation annotation
    plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
             transform=ax.transAxes, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    setup_plot(fig, ax, 'Processing Time vs Token Count')
    ax.set_xlabel('Number of Input Tokens')
    ax.set_ylabel('Processing Time (seconds)')
    
    plt.savefig('examples/time_vs_tokens.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResults saved to examples/benchmark_results.json")
    print("Plots saved as time_vs_output.png and time_vs_tokens.png")

if __name__ == '__main__':
    main()
