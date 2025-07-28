import logging
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions

from immersion_whisper.database.setup import create_db

from .condenser import condense
from .sub_processor import SubtitleBatch

logging.basicConfig()
logger = logging.getLogger("faster_whisper")
logger.setLevel(logging.WARNING)


def extract_audio(input_file: Path, lang: str = "fre") -> Path:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_audio_file:
            temp_audio_path = Path(temp_audio_file.name)
        ffmpeg.input(str(input_file)).output(
            str(temp_audio_path), map=f"0:a:m:language:{lang}", acodec="pcm_s16le"
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        return temp_audio_path
    except ffmpeg.Error:
        return input_file


def format_timestamp(seconds: float) -> str:
    delta = timedelta(seconds=seconds)
    return f"{delta.seconds // 3600:02d}:{delta.seconds // 60 % 60:02d}:{delta.seconds % 60:02d},{delta.microseconds // 1000:03d}"


def transcribe(input_file: Path) -> Path:
    if not input_file.is_file():
        sys.exit(f"Error: Input file not found at {input_file}")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    srt_path = output_dir / f"{input_file.stem}.srt"
    audio_path = extract_audio(input_file)
    is_temp = audio_path != input_file
    episode_number = int(input_file.stem)
    create_db()
    try:
        model = WhisperModel(
            "large-v3-turbo", device="cuda", device_index=0, compute_type="float16"
        )
        vad_params = VadOptions(
            threshold=0.7,
            neg_threshold=0.5,
            min_speech_duration_ms=150,
            max_speech_duration_s=3,
            min_silence_duration_ms=1000,
            speech_pad_ms=200,
        )
        segments_iter, _ = model.transcribe(
            str(audio_path),
            language="fr",
            word_timestamps=True,
            log_progress=True,
            hotwords="Gon, Kirua, Greed Island, Hisoka, Nen, Razor, Jin, Book, Gyo",
            vad_filter=True,
            vad_parameters=vad_params,
        )
        segments = list(segments_iter)
        for seg in segments:
            SubtitleBatch.process_subtitle(
                seg.text.lstrip(), episode_number, seg.start, seg.end
            )
        SubtitleBatch.flush_batch()
        with srt_path.open("w", encoding="utf-8") as srt_file:
            for i, seg in enumerate(segments):
                srt_file.write(
                    f"{i + 1}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.lstrip()}\n\n"
                )
        condense(audio_path, srt_path)
    except Exception as e:
        sys.exit(f"An error occurred: {e}")
    finally:
        if is_temp:
            try:
                Path(audio_path).unlink()
            except Exception:
                pass
    return srt_path
