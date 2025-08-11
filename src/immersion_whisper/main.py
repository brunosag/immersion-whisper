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
from .core.transcriber import transcribe
from .core.translator import translate
from .database.setup import reset_db
from .utils import extract_audio, get_media_files, is_audio

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_pipeline(file_path: Path, output_dir: Path):
    """Runs the main pipeline on one file with the selected steps."""
    audio_path = extract_audio(file_path) if not is_audio(file_path) else file_path
    srt_path = output_dir / f'{file_path.stem}.srt'

    if SETTINGS.pipeline.transcribe:
        transcribe(audio_path, srt_path)

    if SETTINGS.pipeline.translate:
        translated_srt_path = file_path.with_suffix('.srt')
        translate(srt_path, translated_srt_path)

    if SETTINGS.pipeline.condense:
        if not (condensed_dir := os.getenv('CONDENSED_AUDIO_DIR')):
            logger.error(
                'CONDENSED_AUDIO_DIR environment variable is not set. Exiting.'
            )
            sys.exit(1)
        condensed_audio_path = Path(condensed_dir) / f'{srt_path.stem}.mp3'
        condense(audio_path, srt_path, condensed_audio_path)

    if SETTINGS.pipeline.process_subs:
        process_subtitles(srt_path)

    if SETTINGS.pipeline.create_deck:
        deck_name = file_path.stem
        create_deck(file_path, srt_path, deck_name)

    if audio_path != file_path:
        audio_path.unlink(missing_ok=True)


def main():
    args = parse_args()

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    if SETTINGS.pipeline.process_subs:
        reset_db()

    input_path = Path(args.input_path)
    if input_path.is_dir():
        media_file_paths = get_media_files(input_path)
        logger.info("Found %d media files in '%s'.", len(media_file_paths), input_path)
        for file_path in media_file_paths:
            logger.info('---------------- %s ----------------', file_path)
            run_pipeline(file_path, output_dir)
    else:
        run_pipeline(input_path, output_dir)


if __name__ == '__main__':
    main()
