import logging
import re
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions

from .condenser import condense
from .sub_processor import flush_batch, process_subtitle

logging.basicConfig()
logger = logging.getLogger("faster_whisper")
logger.setLevel(logging.WARNING)


def _srt_time_to_seconds(time_str: str) -> float:
    """Converts an SRT time string HH:MM:SS,ms to seconds."""
    hours, minutes, seconds_milliseconds = time_str.split(":")
    seconds, milliseconds = seconds_milliseconds.replace(",", " ").split()
    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    )


def parse_srt_file(srt_path: Path) -> list[dict]:
    """Parses an SRT file and returns a list of subtitle segments."""
    segments = []

    if not srt_path.is_file():
        sys.exit(f"Error: SRT file not found at {srt_path}")

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Regex to find blocks: number, timestamp, and text
    block_pattern = re.compile(
        r"(\d+)\s*?\n"  # Sequence number
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*?-->\s*?(\d{2}:\d{2}:\d{2},\d{3})\s*?\n"  # Timestamps
        r"([\s\S]*?)(?=\n\n|\Z)",  # Subtitle text
        re.MULTILINE,
    )

    for match in block_pattern.finditer(content):
        _, start_time_str, end_time_str, text = match.groups()
        segments.append(
            {
                "start": _srt_time_to_seconds(start_time_str),
                "end": _srt_time_to_seconds(end_time_str),
                "text": text.strip(),
            }
        )
    return segments


def _extract_audio(input_file: Path, lang: str = "fre") -> Path:
    """Extracts audio from the input file using ffmpeg."""
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


def _format_timestamp(seconds: float) -> str:
    """Formats seconds into SRT timestamp format HH:MM:SS,ms."""
    delta = timedelta(seconds=seconds)
    return f"{delta.seconds // 3600:02d}:{delta.seconds // 60 % 60:02d}:{delta.seconds % 60:02d},{delta.microseconds // 1000:03d}"


def transcribe(input_file: Path) -> Path:
    """Transcribes the audio from the input file and generates an SRT file."""
    if not input_file.is_file():
        sys.exit(f"Error: Input file not found at {input_file}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    srt_path = output_dir / f"{input_file.stem}.srt"
    audio_path = _extract_audio(input_file)
    is_temp = audio_path != input_file
    episode_number = int(input_file.stem)
    segments: list[dict] = []

    try:
        if not srt_path.is_file():
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
            print(f"Starting transcription for {input_file}...")
            segments_iter, _ = model.transcribe(
                str(audio_path),
                language="fr",
                word_timestamps=True,
                log_progress=True,
                hotwords="Gon, Kirua, Greed Island, Hisoka, Nen, Razor, Jin, Book, Gyo",
                vad_filter=True,
                vad_parameters=vad_params,
            )
            segments = [
                {"text": seg.text, "start": seg.start, "end": seg.end}
                for seg in segments_iter
            ]
            print(f"Transcription complete. Processing {len(segments)} segments...")
        else:
            print(f"SRT file already exists at {srt_path}. Skipping transcription.")
            segments = parse_srt_file(srt_path)

        for seg in segments:
            process_subtitle(
                seg["text"].lstrip(), episode_number, seg["start"], seg["end"]
            )

        print("Processing complete. Flushing batch...")
        flush_batch()

        print(f"Flushing batch complete. Writing SRT file to {srt_path}...")
        with srt_path.open("w", encoding="utf-8") as srt_file:
            for i, seg in enumerate(segments):
                srt_file.write(
                    f"{i + 1}\n{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}\n{seg['text'].lstrip()}\n\n"
                )

        print("SRT file written successfully. Condensing audio...")
        condense(audio_path, srt_path)
    except Exception as e:
        print(f"Error occurred during transcription: {e}")
    finally:
        if is_temp:
            try:
                Path(audio_path).unlink()
            except Exception:
                pass
    return srt_path
