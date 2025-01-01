#!/usr/bin/env python3
import argparse
from typing import List, Optional

import requests


def submit_combine_voices(voices: List[str], base_url: str = "http://localhost:8880") -> Optional[List[str]]:
    try:
        response = requests.post(f"{base_url}/v1/audio/voices/combine", json=voices)
        if response.status_code != 200:
            print(f"Error submitting request: {response.text}")
            return None
        return response.json()["voices"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Kokoro TTS CLI")
    parser.add_argument("--voices", nargs="+", type=str, help="Voices to combine")
    parser.add_argument("--url", default="http://localhost:8880", help="API base URL")
    args = parser.parse_args()

    success = submit_combine_voices(args.voices, args.url)
    if success:
        for voice in success:
            print(f"  {voice}")


if __name__ == "__main__":
    main()
