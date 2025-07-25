import argparse
import re
from collections import Counter

import pandas as pd
import spacy


def parse_srt(file_path):
    """
    Reads an SRT file, removes timestamps and line numbers,
    and returns a single string of clean text.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: The file was not found at '{file_path}'")
        exit()

    text = re.sub(
        r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", "", text
    )
    text = re.sub(r"[\d\n]+", " ", text)
    text = re.sub(r"<.*?>", "", text)
    return text


def analyze_text(file_path, nlp_model):
    """
    Takes a file path and a spaCy model, analyzes
    the text, and prints the word frequency.
    """
    clean_text = parse_srt(file_path)
    doc = nlp_model(clean_text)

    lemmas = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

    word_freq = Counter(lemmas)

    most_common_list = word_freq.most_common(20)
    words, frequencies = zip(*most_common_list)
    df = pd.DataFrame({"Word": words, "Frequency": frequencies})

    print(f"\nTop 20 most frequent lemmas from '{file_path}':")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes the lemma frequency of a French SRT subtitle file."
    )
    parser.add_argument(
        "srt_file", type=str, help="The path to the .srt file to be analyzed."
    )
    args = parser.parse_args()

    nlp = spacy.load("fr_dep_news_trf")
    analyze_text(args.srt_file, nlp)
