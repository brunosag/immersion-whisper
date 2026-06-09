# `immersion-whisper`

A command-line pipeline designed to process audio and video files for language learning and immersion. It automates transcription, translation, audio condensing, subtitle processing, and Anki flashcard deck generation.

## Features

* **Transcription:** Utilizes `faster-whisper` with Voice Activity Detection (VAD) to generate SubRip Subtitle (SRT) files from media inputs.
* **Translation:** Integrates the Google Gemini API to translate SRT files while strictly preserving timing formats and subtitle indexing.
* **Audio condensing:** Generates condensed audio tracks containing only speech segments.
* **Subtitle processing:** Analyzes transcribed text utilizing `spaCy` NLP models and persists linguistic data using a database via `peewee`.
* **Anki deck generation:** Automatically constructs Anki packages (`.apkg`) via `genanki`. The pipeline uses `ffmpeg` to extract specific audio clips and video frames corresponding to each subtitle, creating rich media flashcards.

## Architecture and pipeline

The application is modularized into configurable pipeline stages, managed by `immersion_whisper/main.py` and defined in `config.default.yaml`:

1. `transcribe`: Extracts audio and generates the base SRT file.
2. `translate`: Calls the Gemini API to produce a translated subtitle file.
3. `condense`: Outputs a condensed `.mp3` containing only spoken audio.
4. `process_subs`: Executes NLP tasks on the transcribed text.
5. `create_deck`: Assembles an Anki deck with integrated media assets extracted from the source file.

## Prerequisites

* **Python:** Version 3.12 or higher.
* **FFmpeg:** Must be installed and accessible in the system PATH for media extraction and audio processing.
* **Environment variables:** Requires a `.env` file containing:
  * `GEMINI_API_KEY`: Required for the translation pipeline.
  * `CONDENSED_AUDIO_DIR`: Required for the audio condensing pipeline.

## Installation

The project uses `pyproject.toml` for dependency management. Install the package and its dependencies using a compatible package manager such as `pip` or `uv`:

```bash
git clone git@github.com:brunosag/immersion-whisper.git
cd immersion-whisper
pip install .
```

## Configuration

Default settings are specified in `config.default.yaml`. The configuration controls pipeline execution toggles, model selection for `faster-whisper` (device, compute type, VAD parameters), the Gemini model ID, target languages, and `spaCy` model selections.

## Usage

Installation exposes the `imw` command-line interface.

```bash
imw <input_path>
```

* `<input_path>`: The file path to a single media file (audio or video) or a directory containing multiple media files. If a directory is provided, the tool processes all supported media files within it sequentially.
* Output artifacts (SRT files, Anki `.apkg` packages, and temporary media) are written to a local `output/` directory by default, while translated files are saved alongside the source media.
