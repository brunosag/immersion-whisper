import os
from pathlib import Path

from ..config import SETTINGS


def translate(srt_path: Path, output_path: Path):
    import requests

    if not (api_key := os.getenv('GEMINI_API_KEY')):
        raise ValueError('Error: GEMINI_API_KEY environment variable is not set.')

    input_content = srt_path.read_text(encoding='utf-8')
    api_url = f'https://generativelanguage.googleapis.com/v1beta/models/{SETTINGS.translator.gemini_model_id}:generateContent?key={api_key}'
    src_lang = SETTINGS.transcriber.language
    tgt_lang = SETTINGS.translator.language
    prompt = (
        f'Translate the following {src_lang} subtitle file to {tgt_lang}.\n'
        'IMPORTANT: You must preserve the exact same timing format and subtitle numbers.\n'
        f'Only translate {src_lang} text to {tgt_lang}, keeping all '
        'timestamps and formatting exactly the same.\n'
        f'Here is the {src_lang} subtitle file:\n\n'
        f'{input_content}'
    )
    headers = {'Content-Type': 'application/json'}
    payload = {'contents': [{'parts': [{'text': prompt}]}]}

    print(f"Translating '{srt_path.name}': {src_lang} -> {tgt_lang}...")
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    try:
        data = response.json()
        translated_text = data['candidates'][0]['content']['parts'][0]['text']
        output_path.write_text(translated_text, encoding='utf-8')
        print(f"Translated subtitles saved to '{output_path}'")
    except (KeyError, IndexError) as e:
        print(f'Failed to parse API response. {e}')
        print(f'Full Response:\n{response.text}')
