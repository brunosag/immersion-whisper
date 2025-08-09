import importlib.resources
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import genanki
import pysrt

logger = logging.getLogger(__name__)


@dataclass
class DeckConfig:
    """Configuration for creating Anki decks."""

    image_quality: int = 5
    image_resolution: str = '640x360'
    audio_batch_size: int = 100
    audio_language_code: str = 'fre'  # ISO 639-2 code (e.g., 'fre', 'jpn', 'eng')


def _time_to_seconds(t: pysrt.SubRipTime) -> float:
    """Converts SubRipTime to seconds."""
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000


def _initialize_anki_components(deck_name: str) -> tuple[genanki.Model, genanki.Deck]:
    """Initializes and returns a genanki Model and Deck."""
    logger.info('Initializing Anki components...')

    try:
        template_path = (
            importlib.resources.files('immersion_whisper.exporters') / 'templates'
        )
        qfmt = (template_path / 'front.html').read_text(encoding='utf-8')
        afmt = (template_path / 'back.html').read_text(encoding='utf-8')
        css = (template_path / 'styles.css').read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError):
        logger.warning(
            'Could not find packaged templates. Using default empty templates.'
        )
        qfmt = '{{sentence}}<br>{{word}}'
        afmt = '{{FrontSide}}<hr id=answer>{{explanation}}<br>{{image}}<br>{{sentence_audio}}'
        css = '.card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }'

    model_name = 'Ankigen-Model'
    model_id = hash(model_name) % (1 << 31)
    deck_id = hash(deck_name) % (1 << 31)

    fields = [
        {'name': 'word'},
        {'name': 'sentence'},
        {'name': 'explanation'},
        {'name': 'word_audio'},
        {'name': 'sentence_audio'},
        {'name': 'explanation_audio'},
        {'name': 'image'},
    ]

    template = {
        'name': 'imw',
        'qfmt': qfmt,
        'afmt': afmt,
    }

    model = genanki.Model(
        model_id, model_name, fields=fields, templates=[template], css=css
    )
    deck = genanki.Deck(deck_id, deck_name)

    return model, deck


def _extract_media(
    video_path: Path,
    subs: pysrt.SubRipFile,
    media_dir: Path,
    deck_name: str,
    config: DeckConfig,
):
    """Extracts images and audio clips from the video based on subtitles."""
    logger.info('Starting media extraction...')
    num_subs = len(subs)

    logger.info('Extracting images...')
    for i, sub in enumerate(subs):
        if (i + 1) % 50 == 0:
            logger.info(f'  - Image {i + 1}/{num_subs}')

        image_time = (
            _time_to_seconds(sub.start)
            + (_time_to_seconds(sub.end) - _time_to_seconds(sub.start)) / 2.0
        )
        image_path = media_dir / f'{deck_name}_{i:04d}.jpg'

        command = [
            'ffmpeg',
            '-y',
            '-ss',
            str(image_time),
            '-i',
            str(video_path),
            '-vframes',
            '1',
            '-q:v',
            str(config.image_quality),
            '-s',
            config.image_resolution,
            str(image_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed to extract image for sub {i}: {e.stderr.strip()}')

    logger.info('Extracting audio in batches...')
    audio_batch_size = config.audio_batch_size
    num_batches = (num_subs + audio_batch_size - 1) // audio_batch_size

    for i in range(num_batches):
        batch_start_index = i * audio_batch_size
        batch_end_index = min((i + 1) * audio_batch_size, num_subs)
        batch_subs = subs[batch_start_index:batch_end_index]
        logger.info(f'  - Processing audio batch {i + 1}/{num_batches}...')

        command = ['ffmpeg', '-y', '-i', str(video_path)]

        for j, sub in enumerate(batch_subs):
            sub_index = batch_start_index + j
            start_time = _time_to_seconds(sub.start)
            end_time = _time_to_seconds(sub.end)
            audio_path = media_dir / f'{deck_name}_{sub_index:04d}.mp3'

            command.extend(
                [
                    '-ss',
                    str(start_time),
                    '-to',
                    str(end_time),
                    '-map',
                    f'0:a:m:language:{config.audio_language_code}',
                    '-c:a',
                    'libmp3lame',
                    '-q:a',
                    '4',
                    str(audio_path),
                ]
            )

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                f'Audio batch {i + 1} failed. Please check if the language code '
                f"'{config.audio_language_code}' is correct for this video file. "
                f'Error: {e.stderr.strip()}'
            )

    logger.info('Media extraction complete.')


def _create_anki_notes(
    subs: pysrt.SubRipFile,
    deck_name: str,
    anki_model: genanki.Model,
    anki_deck: genanki.Deck,
    media_dir: Path,
) -> List[str]:
    """Creates Anki notes and returns a list of media files to include."""
    media_files: list[str] = []
    logger.info('Creating Anki notes...')

    model_field_names = [field['name'] for field in anki_model.fields]

    for i, sub in enumerate(subs):
        base_filename = f'{deck_name}_{i:04d}'
        image_filename = f'{base_filename}.jpg'
        audio_filename = f'{base_filename}.mp3'
        image_path = media_dir / image_filename
        audio_path = media_dir / audio_filename

        if image_path.exists() and audio_path.exists():
            field_content = {
                'word': sub.text.split()[0] if sub.text else '',
                'sentence': sub.text,
                'explanation': '',  # To be filled by another process later
                'word_audio': '',  # To be filled by another process later
                'explanation_audio': '',  # To be filled by another process later
                'sentence_audio': f'[sound:{audio_filename}]',
                'image': f'<img src="{image_filename}">',
            }

            ordered_fields = [field_content.get(name, '') for name in model_field_names]

            note = genanki.Note(model=anki_model, fields=ordered_fields)
            anki_deck.add_note(note)
            media_files.extend([str(image_path), str(audio_path)])

    return media_files


def _generate_anki_package(
    anki_deck: genanki.Deck, media_files: List[str], output_path: Path
):
    """Generates and writes the .apkg file."""
    logger.info('Generating Anki deck package...')
    anki_package = genanki.Package(anki_deck)
    anki_package.media_files = media_files
    anki_package.write_to_file(output_path)


def create_deck(video_path: Path, srt_path: Path, deck_name: str) -> Path | None:
    """Creates an Anki deck from a video and subtitle file."""
    output_dir = Path('output')
    output_deck_path = output_dir / f'{deck_name}.apkg'
    media_dir = output_dir / 'media'

    if not video_path.exists() or not srt_path.exists():
        logger.error(f'Input file not found. Checked {video_path} and {srt_path}')
        return None

    if media_dir.exists():
        logger.info(f'Removing existing media directory: {media_dir}')
        shutil.rmtree(media_dir)
    logger.info(f'Creating fresh media directory: {media_dir}')
    media_dir.mkdir(parents=True, exist_ok=True)

    anki_model, anki_deck = _initialize_anki_components(deck_name)

    logger.info('Parsing subtitles...')
    subs = pysrt.open(str(srt_path), encoding='utf-8')

    _extract_media(video_path, subs, media_dir, deck_name, DeckConfig())

    media_files = _create_anki_notes(subs, deck_name, anki_model, anki_deck, media_dir)

    if anki_deck.notes:
        _generate_anki_package(anki_deck, media_files, output_deck_path)
        logger.info(f"\nSuccessfully created '{output_deck_path}'!")
        logger.info(f"Media files are in '{media_dir}'. This folder can be removed.")
        return output_deck_path
    else:
        logger.error('No notes were generated. The Anki deck could not be created.')
        return None
