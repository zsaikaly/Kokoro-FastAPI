#!/usr/bin/env python3
import argparse
import requests
import time
import sys
import os
from typing import Optional, Tuple


def get_voices(
    base_url: str = "http://localhost:8880",
) -> Optional[Tuple[list[str], str]]:
    """Get list of available voices and default voice"""
    try:
        response = requests.get(f"{base_url}/tts/voices")
        if response.status_code == 200:
            data = response.json()
            return data["voices"], data["default"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting voices: {e}")
    return None


def submit_tts_request(
    text: str, voice: Optional[str] = None, speed: Optional[float] = 1.0, base_url: str = "http://localhost:8880"
) -> Optional[int]:
    """Submit a TTS request and return the request ID"""
    try:
        payload = {"text": text, "speed": speed, "voice": voice} if voice else {"text": text, "speed": speed}
        response = requests.post(f"{base_url}/tts", json=payload)
        if response.status_code != 200:
            print(f"Error submitting request: {response.text}")
            return None
        return response.json()["request_id"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def check_request_status(
    request_id: int, base_url: str = "http://localhost:8880"
) -> Optional[dict]:
    """Check the status of a request"""
    try:
        response = requests.get(f"{base_url}/tts/{request_id}")
        if response.status_code != 200:
            print(f"Error checking status: {response.text}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def download_audio(
    request_id: int, base_url: str = "http://localhost:8880"
) -> Optional[str]:
    """Download and save the generated audio file. Returns the filepath if successful."""
    try:
        response = requests.get(f"{base_url}/tts/file/{request_id}")
        if response.status_code != 200:
            print("Error downloading file")
            return None

        filename = (
            response.headers.get("content-disposition", "")
            .split("filename=")[-1]
            .strip('"')
        )
        if not filename:
            filename = f"speech_{request_id}.wav"

        filepath = os.path.join(os.path.dirname(__file__), "output", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(response.content)
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def generate_speech(
    text: str,
    voice: Optional[str] = None,
    speed: Optional[float] = 1.0,
    base_url: str = "http://localhost:8880",
    download: bool = True,
) -> bool:
    """Generate speech from text"""
    # Submit request
    print("Submitting request...")
    request_id = submit_tts_request(text, voice, speed, base_url)
    if not request_id:
        return False

    print(f"Request submitted (ID: {request_id})")

    # Poll for completion
    while True:
        status = check_request_status(request_id, base_url)
        if not status:
            return False

        if status["status"] == "completed":
            print("Generation complete!")
            if status["processing_time"]:
                print(f"Processing time: {status['processing_time']:.2f}s")

            # Show output file path (clean up any relative path components)
            output_file = status["output_file"]
            if output_file:
                output_file = os.path.normpath(output_file)
            print(f"Output file: {output_file}")

            # Download if requested
            if download:
                print("Downloading file...")
                filepath = download_audio(request_id, base_url)
                if filepath:
                    print(f"Saved to: {filepath}")
                    return True
                return False
            return True

        elif status["status"] == "failed":
            print("Generation failed")
            return False

        print(".", end="", flush=True)
        time.sleep(1)


def list_available_voices(url: str):
    """List all available voices"""
    voices = get_voices(url)
    if voices:
        voices_list, default_voice = voices
        print("Available voices:")
        for voice in voices_list:
            if voice == default_voice:
                print(f"  {voice} (default)")
            else:
                print(f"  {voice}")
    else:
        print("Error getting voices")


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS CLI")
    parser.add_argument("text", nargs="?", help="Text to convert to speech")
    parser.add_argument("--voice", help="Voice to use")
    parser.add_argument("--speed", default=1.0, help="speed of speech")
    parser.add_argument("--url", default="http://localhost:8880", help="API base URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download the file, just show the filepath",
    )
    args = parser.parse_args()

    if args.debug:
        print(f"Debug: Arguments received: {args}")

    # If no text provided, just list voices
    if not args.text:
        list_available_voices(args.url)
        return

    # Generate speech
    print(f"Generating speech for: {args.text}")
    if args.voice:
        print(f"Using voice: {args.voice}")

    if args.debug:
        print(
            f"Debug: Calling generate_speech with text='{args.text}', voice='{args.voice}'"
        )

    success = generate_speech(
        args.text, args.voice, args.speed, args.url, download=not args.no_download
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
