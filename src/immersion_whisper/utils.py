import mimetypes
import tempfile
from pathlib import Path


def extract_audio(input_path: Path, lang: str = 'fre') -> Path:
    """Extracts audio from the input video file using ffmpeg."""
    import ffmpeg

    try:
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as temp_audio_file:
            temp_audio_path = Path(temp_audio_file.name)

        ffmpeg.input(str(input_path)).output(
            str(temp_audio_path), map=f'0:a:m:language:{lang}', acodec='pcm_s16le'
        ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

        return temp_audio_path
    except ffmpeg.Error:
        return input_path


def is_audio(file_path: Path) -> bool:
    """Determines if the provided file path is an audio based on its MIME type."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type.startswith('audio/')
    return False
