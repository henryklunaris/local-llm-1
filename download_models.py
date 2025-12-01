#!/usr/bin/env python3
"""
Download Supertonic TTS ONNX models from HuggingFace.
Run this script before starting the voice assistant for the first time.
"""

import os
import sys


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print("=" * 50)
    print("Downloading Supertonic TTS models from HuggingFace")
    print("=" * 50)
    print()

    # Download ONNX models and voice styles
    local_dir = os.path.join(os.path.dirname(__file__), "assets")
    
    print(f"Downloading to: {local_dir}")
    print("This may take a few minutes (~260MB)...")
    print()

    snapshot_download(
        repo_id="Supertone/supertonic",
        local_dir=local_dir,
        allow_patterns=["onnx/*", "voice_styles/*"],
    )

    print()
    print("=" * 50)
    print("Download complete!")
    print("=" * 50)
    print()
    print("You can now run the voice assistant:")
    print("  uv run python app.py")
    print()


if __name__ == "__main__":
    main()

