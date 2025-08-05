import importlib.resources
import shutil
import subprocess
import sys
from pathlib import Path

import genanki
import pysrt

IMAGE_QUALITY = '5'
IMAGE_RESOLUTION = '640x360'
AUDIO_BATCH_SIZE = 100


def time_to_seconds(t: pysrt.SubRipTime) -> float:
    return t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000


def initialize_anki_components(deck_name: str):
    print('Initializing Anki components...')

    template_path = importlib.resources.files('.exporters') / 'templates'

    model_name = 'ankigen'
    model_id = hash(model_name) % (1 << 31)
    deck_id = hash(deck_name) % (1 << 31)
    fields = [
        'word',
        'sentence',
        'explanation',
        'word_audio',
        'sentence_audio',
        'explanation_audio',
        'image',
    ]
    template = {
        'name': 'ankigen',
        'qfmt': (template_path / 'front.html').read_text(encoding='utf-8'),
        'afmt': (template_path / 'back.html').read_text(encoding='utf-8'),
    }
    model = genanki.Model(
        model_id,
        model_name,
        fields=[dict(name=name) for name in fields],
        templates=[template],
        css=(template_path / 'styles.css').read_text(encoding='utf-8'),
    )
    deck = genanki.Deck(deck_id, deck_name)

    return model, deck


def extract_media(
    video_path: Path,
    subs: pysrt.SubRipFile,
    media_dir: Path,
    deck_name: str,
):
    print('Starting media extraction...')
    num_subs = len(subs)

    print('--- Extracting images individually (fast seek) ---')
    for i, sub in enumerate(subs):
        if (i + 1) % 50 == 0:
            print(f'  - Image {i + 1}/{num_subs}')

        start_time = time_to_seconds(sub.start)
        end_time = time_to_seconds(sub.end)
        image_time = start_time + (end_time - start_time) / 2.0
        image_path = media_dir / f'{deck_name}_{i:04d}.jpg'

        subprocess.run(
            [
                'ffmpeg',
                '-ss',
                str(image_time),
                '-i',
                str(video_path),
                '-vframes',
                '1',
                '-q:v',
                IMAGE_QUALITY,
                '-s',
                IMAGE_RESOLUTION,
                str(image_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    print('\n--- Extracting audio in batches ---')
    num_batches = (num_subs + AUDIO_BATCH_SIZE - 1) // AUDIO_BATCH_SIZE
    for i in range(num_batches):
        batch_start_index = i * AUDIO_BATCH_SIZE
        batch_end_index = min((i + 1) * AUDIO_BATCH_SIZE, num_subs)
        batch_subs = subs[batch_start_index:batch_end_index]
        print(f'  - Processing audio batch {i + 1}/{num_batches}...')

        audio_cmd = ['ffmpeg', '-y', '-i', str(video_path)]
        for j, sub in enumerate(batch_subs):
            sub_index = batch_start_index + j
            start_time = time_to_seconds(sub.start)
            end_time = time_to_seconds(sub.end)
            audio_path = media_dir / f'{deck_name}_{sub_index:04d}.mp3'
            audio_cmd.extend(
                [
                    '-ss',
                    str(start_time),
                    '-to',
                    str(end_time),
                    '-map',
                    '0:a:m:language:fre',
                    '-c:a',
                    'libmp3lame',
                    '-q:a',
                    '4',
                    str(audio_path),
                ]
            )
        subprocess.run(audio_cmd, capture_output=True, text=True, check=False)

    print('\nMedia extraction complete.')


def create_anki_notes(
    subs: pysrt.SubRipFile,
    deck_name: str,
    anki_model: genanki.Model,
    anki_deck: genanki.Deck,
    media_dir: Path,
):
    media_files: list[str] = []
    print('Creating Anki notes...')
    for i, sub in enumerate(subs):
        base_filename = f'{deck_name}_{i:04d}'
        image_filename = f'{base_filename}.jpg'
        audio_filename = f'{base_filename}.mp3'
        image_path = media_dir / image_filename
        audio_path = media_dir / audio_filename

        if image_path.exists() and audio_path.exists():
            note = genanki.Note(
                model=anki_model,
                fields=[
                    sub.text,
                    f'<img src="{image_filename}">',
                    f'[sound:{audio_filename}]',
                ],
            )
            anki_deck.add_note(note)
            media_files.extend([str(image_path), str(audio_path)])
    return media_files


def generate_anki_package(
    anki_deck: genanki.Deck, media_files: list[str], output_path: str | Path
):
    print('\nGenerating Anki deck package...')
    anki_package = genanki.Package(anki_deck)
    anki_package.media_files = media_files
    anki_package.write_to_file(output_path)


def main(video_path: str | Path, srt_path: str | Path, deck_name: str):
    video_file = Path(video_path)
    srt_file = Path(srt_path)
    output_deck_path = Path(f'output/{deck_name}.apkg')
    media_dir = Path('output/media')

    if not video_file.exists() or not srt_file.exists():
        print(f'Error: Input file not found. Checked {video_file} and {srt_file}')
        sys.exit(1)

    if media_dir.exists():
        print(f'Removing existing media directory: {media_dir}')
        shutil.rmtree(media_dir)
    print(f'Creating fresh media directory: {media_dir}')
    media_dir.mkdir()

    anki_model, anki_deck = initialize_anki_components(deck_name)

    print('Parsing subtitles...')
    subs = pysrt.open(str(srt_file))

    extract_media(Path(video_path), subs, media_dir, deck_name)

    media_files = create_anki_notes(subs, deck_name, anki_model, anki_deck, media_dir)

    if anki_deck.notes:
        generate_anki_package(anki_deck, media_files, output_deck_path)
        print(f"\nSuccessfully created '{output_deck_path}'!")
        print(
            f"Media files are in '{media_dir}'. You can remove this folder if you wish."
        )
    else:
        print('\n❌ No notes were generated. The Anki deck could not be created.')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(
            'Usage: python deck_creator.py <video_file.mkv> <subtitles.srt> <DeckName>'
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
