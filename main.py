import argparse
from pathlib import Path

from dotenv import load_dotenv

from transcriber import transcribe
from translator import translate

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file using faster-whisper and generate an SRT file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the video or audio file for transcription.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-large-v3",
            "large-v3-turbo",
            "turbo",
        ],
        default="large-v3-turbo",
        help="Size of the model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="Device to use for computation.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        choices=["float16", "float32", "int8", "auto"],
        default="float16",
        help="Type to use for computation.",
    )
    args = parser.parse_args()

    srt_path = transcribe(
        args.input_file, args.model_size, args.device, args.compute_type
    )
    translate(srt_path, args.input_file)


if __name__ == "__main__":
    main()
