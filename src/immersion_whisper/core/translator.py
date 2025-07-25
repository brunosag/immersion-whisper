import os
from pathlib import Path

import requests


def translate(srt_path: Path, video_path: Path):
    if not (api_key := os.getenv("GEMINI_API_KEY")):
        raise ValueError("Error: GEMINI_API_KEY environment variable is not set.")

    input_content = srt_path.read_text(encoding="utf-8")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent?key={api_key}"
    prompt = (
        "Translate the following French subtitle file to English.\n"
        "IMPORTANT: You must preserve the exact same timing format and subtitle numbers.\n"
        "Only translate the French text to English, keeping all timestamps and "
        "formatting exactly the same.\n"
        "Here is the French subtitle file:\n\n"
        f"{input_content}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    print(f"Translating '{srt_path.name}'...")
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    try:
        data = response.json()
        translated_text = data["candidates"][0]["content"]["parts"][0]["text"]
        output_path = video_path.with_suffix(".srt")
        output_path.write_text(translated_text, encoding="utf-8")
        print(f"Translated subtitles saved to '{output_path}'")
    except (KeyError, IndexError) as e:
        print(f"Failed to parse API response. {e}")
        print(f"Full Response:\n{response.text}")
