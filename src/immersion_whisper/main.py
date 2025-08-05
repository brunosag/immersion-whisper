import os
import sys
import tempfile
from pathlib import Path

import ffmpeg
from dotenv import load_dotenv

from .args import parse_args
from .config import SETTINGS
from .core.condenser import condense
from .core.sub_processor import process_subtitles
from .core.transcriber import transcribe
from .core.translator import translate

load_dotenv()


def extract_audio(input_path: Path, lang: str = 'fre') -> Path:
    """Extracts audio from the input video file using ffmpeg."""
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as temp_audio_file:
            temp_audio_path = Path(temp_audio_file.name)

        ffmpeg.input(str(input_path)).output(
            str(temp_audio_path), map=f'0:a:m:language:{lang}', acodec='pcm_s16le'
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

        return temp_audio_path
    except ffmpeg.Error:
        return input_path


def main():
    args = parse_args()

    audio_path = extract_audio(args.input_file)

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    if SETTINGS.pipeline.transcribe:
        srt_path = output_dir / f'{args.input_file.stem}.srt'
        transcribe(audio_path, srt_path)

    if SETTINGS.pipeline.translate:
        translate_path = args.input_file.with_suffix('.srt')
        translate(srt_path, translate_path)

    if SETTINGS.pipeline.condense:
        if not (OUTPUT_DIR := os.getenv('CONDENSED_AUDIO_DIR')):
            print('CONDENSED_AUDIO_DIR environment variable is not set. Exiting.')
            sys.exit(1)
        condense_path = Path(OUTPUT_DIR) / (srt_path.stem + '.mp3')
        condense(audio_path, srt_path, condense_path)

    if SETTINGS.pipeline.process_subs:
        process_subtitles(srt_path)

    if audio_path != args.input_file:
        audio_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
