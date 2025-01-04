import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

def validate_tts(wav_path: str) -> dict:
    """
    Quick validation checks for TTS-generated audio files to detect common artifacts.
    
    Checks for:
    - Unnatural silence gaps
    - Audio glitches and artifacts
    - Repeated speech segments (stuck/looping)
    - Abrupt changes in speech
    - Audio quality issues
    
    Args:
        wav_path: Path to audio file (wav, mp3, etc)
    Returns:
        Dictionary with validation results
    """
    try:
        # Load audio
        audio, sr = sf.read(wav_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
            
        # Basic audio stats
        duration = len(audio) / sr
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        dc_offset = np.mean(audio)
        
        # Calculate clipping stats if we're near peak
        clip_count = np.sum(np.abs(audio) >= 0.99)
        clip_percent = (clip_count / len(audio)) * 100
        if clip_percent > 0:
            clip_stats = f" ({clip_percent:.2e} ratio near peak)"
        else:
            clip_stats = " (no samples near peak)"
        
        # Convert to dB for analysis
        eps = np.finfo(float).eps
        db = 20 * np.log10(np.abs(audio) + eps)
        
        issues = []
        
        # Check if audio is too short (likely failed generation)
        if duration < 0.1:  # Less than 100ms
            issues.append("WARNING: Audio is suspiciously short - possible failed generation")
        
        # 1. Check for basic audio quality
        if peak >= 1.0:
            # Calculate percentage of samples that are clipping
            clip_count = np.sum(np.abs(audio) >= 0.99)
            clip_percent = (clip_count / len(audio)) * 100
            
            if clip_percent > 1.0:  # Only warn if more than 1% of samples clip
                issues.append(f"WARNING: Significant clipping detected ({clip_percent:.2e}% of samples)")
            elif clip_percent > 0.01:  # Add info if more than 0.01% but less than 1%
                issues.append(f"INFO: Minor peak limiting detected ({clip_percent:.2e}% of samples) - likely intentional normalization")
            
        if rms < 0.01:
            issues.append("WARNING: Audio is very quiet - possible failed generation")
        if abs(dc_offset) > 0.1:  # DC offset is particularly bad for speech
            issues.append(f"WARNING: High DC offset ({dc_offset:.3f}) - may cause audio artifacts")
            
        # 2. Check for long silence gaps (potential TTS failures)
        silence_threshold = -45  # dB
        min_silence = 2.0  # Only detect silences longer than 2 seconds
        window_size = int(min_silence * sr)
        silence_count = 0
        last_silence = -1
        
        # Skip the first 0.2s for silence detection (avoid false positives at start)
        start_idx = int(0.2 * sr)
        for i in range(start_idx, len(db) - window_size, window_size):
            window = db[i:i+window_size]
            if np.mean(window) < silence_threshold:
                # Verify the entire window is mostly silence
                silent_ratio = np.mean(window < silence_threshold)
                if silent_ratio > 0.9:  # 90% of the window should be below threshold
                    if last_silence == -1 or (i/sr - last_silence) > 2.0:  # Only count silences more than 2s apart
                        silence_count += 1
                        last_silence = i/sr
                        issues.append(f"WARNING: Long silence detected at {i/sr:.2f}s (duration: {min_silence:.1f}s)")
        
        if silence_count > 2:  # Only warn if there are multiple long silences
            issues.append(f"WARNING: Multiple long silences found ({silence_count} total) - possible generation issue")
                
        # 3. Check for extreme audio artifacts (changes too rapid for natural speech)
        # Use a longer window to avoid flagging normal phoneme transitions
        window_size = int(0.02 * sr)  # 20ms window
        db_smooth = np.convolve(db, np.ones(window_size)/window_size, 'same')
        db_diff = np.abs(np.diff(db_smooth))
        
        # Much higher threshold to only catch truly unnatural changes
        artifact_threshold = 40  # dB
        min_duration = int(0.01 * sr)  # Minimum 10ms duration
        
        # Find regions where the smoothed dB change is extreme
        artifact_points = np.where(db_diff > artifact_threshold)[0]
        
        if len(artifact_points) > 0:
            # Group artifacts that are very close together
            grouped_artifacts = []
            current_group = [artifact_points[0]]
            
            for i in range(1, len(artifact_points)):
                if (artifact_points[i] - current_group[-1]) < min_duration:
                    current_group.append(artifact_points[i])
                else:
                    if len(current_group) * (1/sr) >= 0.01:  # Only keep groups lasting >= 10ms
                        grouped_artifacts.append(current_group)
                    current_group = [artifact_points[i]]
            
            if len(current_group) * (1/sr) >= 0.01:
                grouped_artifacts.append(current_group)
            
            # Report only the most severe artifacts
            for group in grouped_artifacts[:2]:  # Report up to 2 worst artifacts
                center_idx = group[len(group)//2]
                db_change = db_diff[center_idx]
                if db_change > 45:  # Only report very extreme changes
                    issues.append(
                        f"WARNING: Possible audio artifact at {center_idx/sr:.2f}s "
                        f"({db_change:.1f}dB change over {len(group)/sr*1000:.0f}ms)"
                    )
            
        # 4. Check for repeated speech segments (stuck/looping)
        # Check both short and long sentence durations at audiobook speed (150-160 wpm)
        for chunk_duration in [5.0, 10.0]:  # 5s (~12 words) and 10s (~25 words) at ~audiobook speed
            chunk_size = int(chunk_duration * sr)
            overlap = int(0.2 * chunk_size)  # 20% overlap between chunks
            
            for i in range(0, len(audio) - 2*chunk_size, overlap):
                chunk1 = audio[i:i+chunk_size]
                chunk2 = audio[i+chunk_size:i+2*chunk_size]
                
                # Ignore chunks that are mostly silence
                if np.mean(np.abs(chunk1)) < 0.01 or np.mean(np.abs(chunk2)) < 0.01:
                    continue
                    
                try:
                    correlation = np.corrcoef(chunk1, chunk2)[0,1]
                    if not np.isnan(correlation) and correlation > 0.92:  # Lower threshold for sentence-length chunks
                        issues.append(
                            f"WARNING: Possible repeated speech at {i/sr:.1f}s "
                            f"(~{int(chunk_duration*160/60):d} words, correlation: {correlation:.3f})"
                        )
                        break  # Found repetition at this duration, try next duration
                except:
                    continue
        
        # 5. Check for extreme amplitude discontinuities (common in failed TTS)
        amplitude_envelope = np.abs(audio)
        window_size = sr // 10  # 100ms window for smoother envelope
        smooth_env = np.convolve(amplitude_envelope, np.ones(window_size)/float(window_size), 'same')
        env_diff = np.abs(np.diff(smooth_env))
        
        # Only detect very extreme amplitude changes
        jump_threshold = 0.5  # Much higher threshold
        jumps = np.where(env_diff > jump_threshold)[0]
        
        if len(jumps) > 0:
            # Group jumps that are close together
            grouped_jumps = []
            current_group = [jumps[0]]
            
            for i in range(1, len(jumps)):
                if (jumps[i] - current_group[-1]) < 0.05 * sr:  # Group within 50ms
                    current_group.append(jumps[i])
                else:
                    if len(current_group) >= 3:  # Only keep significant discontinuities
                        grouped_jumps.append(current_group)
                    current_group = [jumps[i]]
            
            if len(current_group) >= 3:
                grouped_jumps.append(current_group)
            
            # Report only the most severe discontinuities
            for group in grouped_jumps[:2]:  # Report up to 2 worst cases
                center_idx = group[len(group)//2]
                jump_size = env_diff[center_idx]
                if jump_size > 0.6:  # Only report very extreme changes
                    issues.append(
                        f"WARNING: Possible audio discontinuity at {center_idx/sr:.2f}s "
                        f"({jump_size:.2f} amplitude ratio change)"
                    )
        
        return {
            "file": wav_path,
            "duration": f"{duration:.2f}s",
            "sample_rate": sr,
            "peak_amplitude": f"{peak:.3f}{clip_stats}",
            "rms_level": f"{rms:.3f}",
            "dc_offset": f"{dc_offset:.3f}",
            "issues": issues,
            "valid": len(issues) == 0
        }
        
    except Exception as e:
        return {
            "file": wav_path,
            "error": str(e),
            "valid": False
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Output Validator")
    parser.add_argument("wav_file", help="Path to audio file to validate")
    args = parser.parse_args()
    
    result = validate_tts(args.wav_file)
    
    print(f"\nValidating: {result['file']}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Duration: {result['duration']}")
        print(f"Sample Rate: {result['sample_rate']} Hz")
        print(f"Peak Amplitude: {result['peak_amplitude']}")
        print(f"RMS Level: {result['rms_level']}")
        print(f"DC Offset: {result['dc_offset']}")
        
        if result["issues"]:
            print("\nIssues Found:")
            for issue in result["issues"]:
                print(f"- {issue}")
        else:
            print("\nNo issues found")
