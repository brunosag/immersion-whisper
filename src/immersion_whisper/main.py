import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .args import parse_args
from .config import SETTINGS
from .core.condenser import condense
from .core.deck_creator import create_deck
from .core.sub_processor import process_subtitles
from .core.transcriber import extract_audio, transcribe
from .core.translator import translate

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    args = parse_args()

    input_file_path = args.input_file
    audio_path = extract_audio(input_file_path)
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    srt_path = output_dir / f'{input_file_path.stem}.srt'

    if SETTINGS.pipeline.transcribe:
        transcribe(audio_path, srt_path)

    if SETTINGS.pipeline.translate:
        translated_srt_path = input_file_path.with_suffix('.srt')
        translate(srt_path, translated_srt_path)

    if SETTINGS.pipeline.condense:
        if not (condensed_dir := os.getenv('CONDENSED_AUDIO_DIR')):
            logging.error(
                'CONDENSED_AUDIO_DIR environment variable is not set. Exiting.'
            )
            sys.exit(1)
        condensed_audio_path = Path(condensed_dir) / f'{srt_path.stem}.mp3'
        condense(audio_path, srt_path, condensed_audio_path)

    if SETTINGS.pipeline.process_subs:
        process_subtitles(srt_path)

    if SETTINGS.pipeline.create_deck:
        deck_name = input_file_path.stem
        create_deck(input_file_path, srt_path, deck_name)

    if audio_path != args.input_file:
        audio_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
