import logging
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import ffmpeg
from faster_whisper import WhisperModel
from faster_whisper.transcribe import VadOptions

from .condenser import condense

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def extract_audio_stream(input_file: Path, language_code: str = "fre"):
    try:
        print(f"Checking for '{language_code}' audio stream in {input_file.name}...")
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = Path(temp_audio_file.name)
        temp_audio_file.close()

        (
            ffmpeg.input(str(input_file))
            .output(
                str(temp_audio_path),
                map=f"0:a:m:language:{language_code}",
                acodec="pcm_s16le",
            )
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        print(
            f"Successfully extracted '{language_code}' audio to temporary file: {temp_audio_path}"
        )
        return temp_audio_path
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode()}", file=sys.stderr)
        print(
            "Could not find a specific French audio track. Using the default stream.",
            file=sys.stderr,
        )
        return input_file


def format_timestamp(seconds: float):
    delta = timedelta(seconds=seconds)
    return f"{delta.seconds // 3600:02d}:{delta.seconds // 60 % 60:02d}:{delta.seconds % 60:02d},{delta.microseconds // 1000:03d}"


def transcribe(
    input_file: Path,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "auto",
):
    if not input_file.is_file():
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    srt_path = output_dir / (input_file.stem + ".srt")
    processing_file_path = extract_audio_stream(input_file, language_code="fre")
    is_temp_file = processing_file_path != input_file

    try:
        model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            device_index=0,
            compute_type="float16",
        )  # type: ignore

        vad_parameters = VadOptions(
            threshold=0.65,
            neg_threshold=0.55,
            min_speech_duration_ms=100,
            max_speech_duration_s=3,
            min_silence_duration_ms=250,
            speech_pad_ms=700,
        )

        segments, _ = model.transcribe(
            str(processing_file_path),
            language="fr",
            word_timestamps=True,
            log_progress=True,
            hotwords="Gon, Kirua, Greed Island, Hisoka, Nen, Razor, Jin, Book, Gyo",
            vad_filter=True,
            vad_parameters=vad_parameters,
        )

        with srt_path.open("w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(segments):
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                text = segment.text.lstrip()

                srt_file.write(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")
        print(f"srt file saved to: {srt_path}")

        condense(processing_file_path, srt_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        if is_temp_file:
            print(f"Cleaning up temporary file: {processing_file_path}")
            Path(processing_file_path).unlink()
    return srt_path
