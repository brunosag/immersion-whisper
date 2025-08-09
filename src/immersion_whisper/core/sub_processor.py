import logging
import random
import re
import sys
from pathlib import Path

import pandas as pd
import spacy

from ..config import SETTINGS
from ..database.models import Lemma, Subtitle, SubtitleLemma, db
from ..database.setup import reset_db

logger = logging.getLogger(__name__)

_NLP_MODEL = None


def _get_nlp():
    """Lazily loads the spaCy model."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        _NLP_MODEL = spacy.load(
            SETTINGS.sub_processor.spacy_model, disable=['parser', 'ner']
        )
    return _NLP_MODEL


def parse_srt_file(srt_path: Path) -> list[dict]:
    """Parses an SRT file and returns a list of subtitle segments."""
    segments = []

    if not srt_path.is_file():
        logger.error('SRT file not found at %s', srt_path)
        sys.exit(1)

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Regex to find blocks: number, timestamp, and text
    block_pattern = re.compile(
        r'(\d+)\s*?\n'  # Sequence number
        r'(\d{2}:\d{2}:\d{2},\d{3})\s*?-->\s*?(\d{2}:\d{2}:\d{2},\d{3})\s*?\n'  # Timestamps
        r'([\s\S]*?)(?=\n\n|\Z)',  # Subtitle text
        re.MULTILINE,
    )

    for match in block_pattern.finditer(content):
        _, start_time_str, end_time_str, text = match.groups()
        segments.append(
            {
                'start': srt_time_to_seconds(start_time_str),
                'end': srt_time_to_seconds(end_time_str),
                'text': text.strip(),
            }
        )
    return segments


def srt_time_to_seconds(time_str: str) -> float:
    """Converts an SRT time string HH:MM:SS,ms to seconds."""
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.replace(',', ' ').split()
    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    )


class SubtitleProcessor:
    def __init__(self):
        self.subtitles_data: list[dict] = []
        self.lemma_cache: dict[str, int] | None = None

    def _load_cache_if_needed(self):
        """Loads the lemma cache, assuming the database and tables already exist."""
        if self.lemma_cache is None:
            try:
                self.lemma_cache = {
                    lemma.text: lemma.id for lemma in Lemma.select(Lemma.id, Lemma.text)
                }
            except Exception as e:
                logger.error(f'Error loading lemma cache: {e}')
                raise

    def add(self, text: str, episode: int, start: float, end: float):
        """Adds raw subtitle data to the internal storage."""
        self.subtitles_data.append(
            {
                'text': text,
                'episode_number': episode,
                'starts_at': round(start, 3),
                'ends_at': round(end, 3),
            }
        )

    def _lemmatize_batch(self, texts: list[str]) -> list[list[str]]:
        """Lemmatizes a batch of texts."""
        nlp = _get_nlp()
        lemmas_list: list[list[str]] = []
        for doc in nlp.pipe(texts):
            lemmas_list.append(
                [
                    token.lemma_.lower()
                    for token in doc
                    if token.is_alpha and not token.is_stop and not token.is_punct
                ]
            )
        return lemmas_list

    def process(self):
        """Processes the entire batch of subtitles."""
        self._load_cache_if_needed()
        logger.info('Processing subtitle batch...')

        if not self.subtitles_data or self.lemma_cache is None:
            return

        try:
            df = pd.DataFrame(self.subtitles_data)
            df['lemmas'] = self._lemmatize_batch(df['text'].tolist())

            with db.atomic():
                # Identify and insert new lemmas
                exploded_df = df.explode('lemmas').dropna(subset=['lemmas'])
                all_lemmas_in_batch = set(exploded_df['lemmas'])
                new_lemma_texts = all_lemmas_in_batch - self.lemma_cache.keys()

                if new_lemma_texts:
                    new_lemma_records = [{'text': t} for t in new_lemma_texts]
                    Lemma.insert_many(new_lemma_records).execute()

                    # Update cache with the new lemmas
                    newly_added = Lemma.select().where(Lemma.text.in_(new_lemma_texts))
                    for lemma in newly_added:
                        self.lemma_cache[lemma.text] = lemma.id

                # Insert subtitles
                sub_records = df[
                    ['text', 'episode_number', 'starts_at', 'ends_at']
                ].to_dict('records')
                if not sub_records:
                    return
                Subtitle.insert_many(sub_records).execute()

                # Fetch inserted subtitles to get their DB IDs
                fetched_subs = Subtitle.select().where(
                    Subtitle.text.in_(df['text'].unique().tolist())
                )
                sub_id_map = {
                    (s.text, s.episode_number, s.starts_at, s.ends_at): s.id
                    for s in fetched_subs
                }
                df['subtitle_id'] = df.apply(
                    lambda row: sub_id_map.get(
                        (
                            row['text'],
                            row['episode_number'],
                            row['starts_at'],
                            row['ends_at'],
                        )
                    ),  # type: ignore
                    axis=1,
                )

                # Prepare and insert many-to-many relationships using the DataFrame
                rels_df = df.explode('lemmas').dropna(subset=['lemmas', 'subtitle_id'])
                if not rels_df.empty:
                    rels_df['lemma_id'] = rels_df['lemmas'].map(self.lemma_cache)
                    rels_to_insert = [
                        {'subtitle': r['subtitle_id'], 'lemma': r['lemma_id']}
                        for r in rels_df[['subtitle_id', 'lemma_id']].to_dict('records')
                    ]
                    if rels_to_insert:
                        SubtitleLemma.insert_many(rels_to_insert).execute()

                # Select a random associated subtitle for every lemma in the batch
                lemmas_in_batch_ids = {
                    self.lemma_cache[lemma_text] for lemma_text in all_lemmas_in_batch
                }
                lemma_to_subtitles_map = (
                    rels_df.groupby('lemma_id')['subtitle_id'].apply(list).to_dict()
                )
                lemmas_to_update = [
                    Lemma(
                        id=lemma_id, card_subtitle=random.choice(associated_subtitles)
                    )
                    for lemma_id in lemmas_in_batch_ids
                    if (associated_subtitles := lemma_to_subtitles_map.get(lemma_id))
                ]
                if lemmas_to_update:
                    Lemma.bulk_update(lemmas_to_update, fields=[Lemma.card_subtitle])

        except Exception as e:
            logger.error(f'Batch processing error: {e}')
            raise
        finally:
            self.subtitles_data.clear()


_processor = SubtitleProcessor()


def process_subtitle(text: str, episode: int, start: float, end: float):
    """Public API to add a subtitle to the batch."""
    _processor.add(text, episode, start, end)


def flush_batch():
    """Public API to process the current batch."""
    _processor.process()


def process_subtitles(srt_path: Path):
    """Processes subtitles from an SRT file and stores them in the database."""
    reset_db()
    episode_number = int(srt_path.stem) if srt_path.stem.isnumeric() else 0
    segments = parse_srt_file(srt_path)
    for segment in segments:
        process_subtitle(
            segment['text'], episode_number, segment['start'], segment['end']
        )
    flush_batch()
    logger.info("Subtitles at '%s' processed and stored.", srt_path)
