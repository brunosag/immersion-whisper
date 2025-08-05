import logging
import sys
from datetime import timedelta
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions

from ..config import SETTINGS

logging.basicConfig()
logger = logging.getLogger('faster_whisper')
logger.setLevel(logging.WARNING)


def _format_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format HH:MM:SS,ms."""
    delta = timedelta(seconds=seconds)
    return f'{delta.seconds // 3600:02d}:{delta.seconds // 60 % 60:02d}:{delta.seconds % 60:02d},{delta.microseconds // 1000:03d}'


def transcribe(audio_path: Path, srt_path: Path):
    """Transcribes the audio from the input file and generates an SRT file."""
    if not audio_path.is_file():
        sys.exit(f"Error: Input audio file not found at '{audio_path}'")
    if srt_path.is_file():
        print(f"SRT file already exists at '{srt_path}'. Skipping transcription.")
        return

    print(f'Loading {SETTINGS.transcriber.model.name} model...')
    model = WhisperModel(
        SETTINGS.transcriber.model.name,
        device=SETTINGS.transcriber.model.device,
        compute_type=SETTINGS.transcriber.model.compute_type,
        device_index=0,
    )
    vad_params = VadOptions(
        threshold=SETTINGS.transcriber.vad.threshold,
        neg_threshold=SETTINGS.transcriber.vad.neg_threshold,
        min_speech_duration_ms=SETTINGS.transcriber.vad.min_speech_duration_ms,
        max_speech_duration_s=SETTINGS.transcriber.vad.max_speech_duration_s,
        min_silence_duration_ms=SETTINGS.transcriber.vad.min_silence_duration_ms,
        speech_pad_ms=SETTINGS.transcriber.vad.speech_pad_ms,
    )
    print(
        f"Starting transcription for '{audio_path}' (VAD: {'ON' if SETTINGS.transcriber.vad.active else 'OFF'})..."
    )
    segments_iter, _ = model.transcribe(
        str(audio_path),
        language=SETTINGS.transcriber.language,
        word_timestamps=True,
        log_progress=True,
        hotwords=', '.join(SETTINGS.transcriber.hotwords),
        vad_filter=SETTINGS.transcriber.vad.active,
        vad_parameters=vad_params if SETTINGS.transcriber.vad.active else None,
        initial_prompt=SETTINGS.transcriber.initial_prompt,
    )
    segments = [
        {'text': seg.text, 'start': seg.start, 'end': seg.end} for seg in segments_iter
    ]
    with srt_path.open('w', encoding='utf-8') as srt_file:
        for i, seg in enumerate(segments):
            srt_file.write(
                f'{i + 1}\n{_format_timestamp(seg["start"])} --> {_format_timestamp(seg["end"])}\n{seg["text"].lstrip()}\n\n'
            )
    print(f"Transcription complete. SRT file saved to '{srt_path}'")
