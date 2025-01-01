#!/usr/bin/env python3
import argparse
import os
from typing import List, Optional, Dict, Tuple

import requests
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def submit_combine_voices(voices: List[str], base_url: str = "http://localhost:8880") -> Optional[str]:
    """Combine multiple voices into a new voice.
    
    Args:
        voices: List of voice names to combine (e.g. ["af_bella", "af_sarah"])
        base_url: API base URL
        
    Returns:
        Name of the combined voice (e.g. "af_bella_af_sarah") or None if error
    """
    try:
        response = requests.post(f"{base_url}/v1/audio/voices/combine", json=voices)
        print(f"Response status: {response.status_code}")
        print(f"Raw response: {response.text}")
        
        # Accept both 200 and 201 as success
        if response.status_code not in [200, 201]:
            try:
                error = response.json()["detail"]["message"]
                print(f"Error combining voices: {error}")
            except:
                print(f"Error combining voices: {response.text}")
            return None
            
        try:
            data = response.json()
            if "voices" in data:
                print(f"Available voices: {', '.join(sorted(data['voices']))}")
            return data["voice"]
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_speech(text: str, voice: str, base_url: str = "http://localhost:8880", output_file: str = "output.mp3") -> bool:
    """Generate speech using specified voice.
    
    Args:
        text: Text to convert to speech
        voice: Voice name to use
        base_url: API base URL
        output_file: Path to save audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "speed": 1.0,
                "response_format": "wav"  # Use WAV for analysis
            }
        )
        
        if response.status_code != 200:
            error = response.json().get("detail", {}).get("message", response.text)
            print(f"Error generating speech: {error}")
            return False
            
        # Save the audio
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Saved audio to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def analyze_audio(filepath: str) -> Tuple[np.ndarray, int, dict]:
    """Analyze audio file and return samples, sample rate, and audio characteristics.
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Tuple of (samples, sample_rate, characteristics)
    """
    sample_rate, samples = wavfile.read(filepath)
    
    # Convert to mono if stereo
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)
    
    # Calculate basic stats
    max_amp = np.max(np.abs(samples))
    rms = np.sqrt(np.mean(samples**2))
    duration = len(samples) / sample_rate
    
    # Zero crossing rate (helps identify voice characteristics)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(samples)))) / len(samples)
    
    # Simple frequency analysis
    if len(samples) > 0:
        # Use FFT to get frequency components
        fft_result = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
        
        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_result)[pos_mask]
        
        # Find dominant frequencies (top 3)
        top_indices = np.argsort(magnitudes)[-3:]
        dominant_freqs = freqs[top_indices]
        
        # Calculate spectral centroid (brightness of sound)
        spectral_centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
    else:
        dominant_freqs = []
        spectral_centroid = 0
    
    characteristics = {
        "max_amplitude": max_amp,
        "rms": rms,
        "duration": duration,
        "zero_crossing_rate": zero_crossings,
        "dominant_frequencies": dominant_freqs,
        "spectral_centroid": spectral_centroid
    }
    
    return samples, sample_rate, characteristics


def setup_plot(fig, ax, title):
    """Configure plot styling"""
    # Improve grid
    ax.grid(True, linestyle="--", alpha=0.3, color="#ffffff")

    # Set title and labels with better fonts
    ax.set_title(title, pad=20, fontsize=16, fontweight="bold", color="#ffffff")
    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="medium", color="#ffffff")
    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="medium", color="#ffffff")

    # Improve tick labels
    ax.tick_params(labelsize=12, colors="#ffffff")

    # Style spines
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_alpha(0.3)
        spine.set_linewidth(0.5)

    # Set background colors
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    return fig, ax

def plot_analysis(audio_files: Dict[str, str], output_dir: str):
    """Plot comprehensive voice analysis including waveforms and metrics comparison.
    
    Args:
        audio_files: Dictionary of label -> filepath
        output_dir: Directory to save plot files
    """
    # Set dark style
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    fig.patch.set_facecolor("#1a1a2e")
    num_files = len(audio_files)
    
    # Create subplot grid with proper spacing
    gs = plt.GridSpec(num_files + 1, 2, height_ratios=[1.5]*num_files + [1], 
                     hspace=0.4, wspace=0.3)
    
    # Analyze all files first
    all_chars = {}
    for i, (label, filepath) in enumerate(audio_files.items()):
        samples, sample_rate, chars = analyze_audio(filepath)
        all_chars[label] = chars
        
        # Plot waveform spanning both columns
        ax = plt.subplot(gs[i, :])
        time = np.arange(len(samples)) / sample_rate
        plt.plot(time, samples / chars['max_amplitude'], linewidth=0.5, color="#ff2a6d")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Normalized Amplitude")
        ax.set_ylim(-1.1, 1.1)
        setup_plot(fig, ax, f"Waveform: {label}")
    
    # Colors for voices
    colors = ["#ff2a6d", "#05d9e8", "#d1f7ff"]
    
    # Create two subplots for metrics with similar scales
    # Left subplot: Brightness and Volume
    ax1 = plt.subplot(gs[num_files, 0])
    metrics1 = [
        ('Brightness', [chars['spectral_centroid']/1000 for chars in all_chars.values()], 'kHz'),
        ('Volume', [chars['rms']*100 for chars in all_chars.values()], 'RMS×100')
    ]
    
    # Right subplot: Voice Pitch and Texture
    ax2 = plt.subplot(gs[num_files, 1])
    metrics2 = [
        ('Voice Pitch', [min(chars['dominant_frequencies']) for chars in all_chars.values()], 'Hz'),
        ('Texture', [chars['zero_crossing_rate']*1000 for chars in all_chars.values()], 'ZCR×1000')
    ]
    
    def plot_grouped_bars(ax, metrics, show_legend=True):
        n_groups = len(metrics)
        n_voices = len(audio_files)
        bar_width = 0.25
        
        indices = np.arange(n_groups)
        
        # Get max value for y-axis scaling
        max_val = max(max(m[1]) for m in metrics)
        
        for i, (voice, color) in enumerate(zip(audio_files.keys(), colors)):
            values = [m[1][i] for m in metrics]
            offset = (i - n_voices/2 + 0.5) * bar_width
            bars = ax.bar(indices + offset, values, bar_width,
                         label=voice, color=color, alpha=0.8)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', color='white',
                       fontsize=10)
        
        ax.set_xticks(indices)
        ax.set_xticklabels([f"{m[0]}\n({m[2]})" for m in metrics])
        
        # Set y-axis limits with some padding
        ax.set_ylim(0, max_val * 1.2)
        
        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     facecolor="#1a1a2e", edgecolor="#ffffff")
    
    # Plot both subplots
    plot_grouped_bars(ax1, metrics1, show_legend=True)
    plot_grouped_bars(ax2, metrics2, show_legend=False)
    
    # Style both subplots
    setup_plot(fig, ax1, 'Brightness and Volume')
    setup_plot(fig, ax2, 'Voice Pitch and Texture')
    
    # Add y-axis labels
    ax1.set_ylabel('Value')
    ax2.set_ylabel('Value')
    
    # Adjust the figure size to accommodate the legend
    fig.set_size_inches(15, 15)
    
    # Add padding around the entire figure
    plt.subplots_adjust(right=0.85, top=0.95, bottom=0.05, left=0.1)
    plt.savefig(os.path.join(output_dir, "analysis_comparison.png"), dpi=300)
    print(f"Saved analysis comparison to {output_dir}/analysis_comparison.png")
    
    # Print detailed comparative analysis
    print("\nDetailed Voice Analysis:")
    for label, chars in all_chars.items():
        print(f"\n{label}:")
        print(f"  Max Amplitude: {chars['max_amplitude']:.2f}")
        print(f"  RMS (loudness): {chars['rms']:.2f}")
        print(f"  Duration: {chars['duration']:.2f}s")
        print(f"  Zero Crossing Rate: {chars['zero_crossing_rate']:.3f}")
        print(f"  Spectral Centroid: {chars['spectral_centroid']:.0f}Hz")
        print(f"  Dominant Frequencies: {', '.join(f'{f:.0f}Hz' for f in chars['dominant_frequencies'])}")


def main():
    parser = argparse.ArgumentParser(description="Kokoro Voice Analysis Demo")
    parser.add_argument("--voices", nargs="+", type=str, help="Voices to combine")
    parser.add_argument("--text", type=str, default="Hello! This is a test of combined voices.", help="Text to speak")
    parser.add_argument("--url", default="http://localhost:8880", help="API base URL")
    parser.add_argument("--output-dir", default="examples/output", help="Output directory for audio files")
    args = parser.parse_args()

    if not args.voices:
        print("No voices provided, using default test voices")
        args.voices = ["af_bella", "af_nicole"]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store audio files for analysis
    audio_files = {}
    
    # Generate speech with individual voices
    print("Generating speech with individual voices...")
    for voice in args.voices:
        output_file = os.path.join(args.output_dir, f"analysis_{voice}.wav")
        if generate_speech(args.text, voice, args.url, output_file):
            audio_files[voice] = output_file
    
    # Generate speech with combined voice
    print(f"\nCombining voices: {', '.join(args.voices)}")
    combined_voice = submit_combine_voices(args.voices, args.url)
    
    if combined_voice:
        print(f"Successfully created combined voice: {combined_voice}")
        output_file = os.path.join(args.output_dir, f"analysis_combined_{combined_voice}.wav")
        if generate_speech(args.text, combined_voice, args.url, output_file):
            audio_files["combined"] = output_file
    
        # Generate comparison plots
        plot_analysis(audio_files, args.output_dir)
    else:
        print("Failed to combine voices")


if __name__ == "__main__":
    main()
