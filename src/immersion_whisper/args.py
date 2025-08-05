import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transcribe an audio/video file using faster-whisper and generate an SRT file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to the video or audio file for transcription.',
    )

    return parser.parse_args()
