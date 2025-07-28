import logging
import random

import pandas as pd
import spacy

from ..database.models import Lemma, Subtitle, SubtitleLemma, db

logger = logging.getLogger(__name__)


class SubtitleBatch:
    _instance = None
    _nlp_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance

    def _init_once(self):
        self.subtitles = []
        self.lemma_cache = {}
        self._load_lemmas()

    @staticmethod
    def get_nlp():
        if SubtitleBatch._nlp_model is None:
            SubtitleBatch._nlp_model = spacy.load("fr_dep_news_trf")
        return SubtitleBatch._nlp_model

    def _load_lemmas(self):
        self.lemma_cache = {lemma.text: lemma.id for lemma in Lemma.select()}

    def add(self, text, episode, start, end):
        lemmas = self.lemmatize(text)
        self.subtitles.append(
            {
                "text": text,
                "episode_number": episode,
                "starts_at": start,
                "ends_at": end,
                "lemmas": lemmas,
            }
        )

    def process(self):
        if not self.subtitles:
            return
        try:
            with db.atomic():
                df = pd.DataFrame(self.subtitles)
                all_lemmas = set(lem for lemma in df["lemmas"] for lem in (lemma or []))
                new_lemmas = [
                    {"text": lemma, "frequency": 0}
                    for lemma in all_lemmas
                    if lemma not in self.lemma_cache
                ]
                if new_lemmas:
                    Lemma.insert_many(new_lemmas).execute()
                    self._load_lemmas()
                subs = df[["text", "episode_number", "starts_at", "ends_at"]].to_dict(
                    "records"
                )
                if subs:
                    Subtitle.insert_many(subs).execute()
                    sub_ids = {
                        (s.text, s.episode_number, s.starts_at, s.ends_at): s.id
                        for s in Subtitle.select().where(
                            Subtitle.text.in_([r["text"] for r in subs])
                        )
                    }
                    rels = [
                        {
                            "subtitle": sub_ids[
                                (
                                    sub["text"],
                                    sub["episode_number"],
                                    sub["starts_at"],
                                    sub["ends_at"],
                                )
                            ],
                            "lemma": self.lemma_cache[lemma],
                        }
                        for sub in self.subtitles
                        for lemma in (sub["lemmas"] or [])
                        if lemma in self.lemma_cache
                    ]
                    if rels:
                        for i in range(0, len(rels), 500):
                            SubtitleLemma.insert_many(rels[i : i + 500]).execute()
                    for lemma_id in self.lemma_cache.values():
                        subtitle_ids = [
                            sl.subtitle.id
                            for sl in SubtitleLemma.select().where(
                                SubtitleLemma.lemma == lemma_id
                            )
                        ]
                        if subtitle_ids:
                            random_sub_id = random.choice(subtitle_ids)
                            Lemma.update(card_subtitle=random_sub_id).where(
                                Lemma.id == lemma_id
                            ).execute()
        except Exception as e:
            logger.error(f"Batch error: {e}")
            raise
        finally:
            self.subtitles.clear()

    @staticmethod
    def process_subtitle(text, episode, start, end):
        SubtitleBatch().add(text, episode, start, end)

    @staticmethod
    def flush_batch():
        SubtitleBatch().process()

    @staticmethod
    def lemmatize(text):
        nlp = SubtitleBatch.get_nlp()
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
