import os
import sys
from pathlib import Path

import pysrt
from pydub import AudioSegment

from ..config import SETTINGS


def srt_time_to_ms(time_obj: pysrt.SubRipTime):
    """Converts a SubRipTime object to milliseconds."""
    return (
        time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds
    ) * 1000 + time_obj.milliseconds


def condense(wav_file: Path, srt_file: Path):
    if not (OUTPUT_DIR := os.getenv('CONDENSED_AUDIO_DIR')):
        print('CONDENSED_AUDIO_DIR environment variable is not set. Exiting.')
        sys.exit(1)

    audio = AudioSegment.from_wav(str(wav_file))
    subs = pysrt.open(str(srt_file))

    if not subs:
        print(f'No subtitles found in the file: {srt_file}. Exiting.')
        sys.exit(1)

    intervals = []
    for sub in subs:
        start = srt_time_to_ms(sub.start)
        end = srt_time_to_ms(sub.end)

        padding_ms = SETTINGS.condenser.padding_ms
        padded_start = max(0, start - padding_ms)
        padded_end = end + padding_ms
        intervals.append([padded_start, padded_end])

    intervals.sort(key=lambda x: x[0])

    merged_intervals = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        _, last_end = merged_intervals[-1]

        if current_start < last_end:
            merged_intervals[-1][1] = max(last_end, current_end)
        else:
            merged_intervals.append([current_start, current_end])

    condensed_audio = AudioSegment.empty()

    for start, end in merged_intervals:
        segment = audio[start:end]
        condensed_audio += segment

    output_file = Path(OUTPUT_DIR) / (srt_file.stem + '.mp3')
    output_file.parent.mkdir(exist_ok=True)
    condensed_audio.export(output_file, format='mp3', parameters=['-q:a', '2'])
    print(f'Condensed audio saved to: {output_file}')
