from pathlib import Path
from typing import Any, MutableMapping

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    device: str
    compute_type: str


class VadConfig(BaseModel):
    active: bool
    threshold: float
    neg_threshold: float
    min_speech_duration_ms: int
    max_speech_duration_s: int
    min_silence_duration_ms: int
    speech_pad_ms: int


class TranscriberConfig(BaseModel):
    language: str
    model: ModelConfig
    vad: VadConfig
    hotwords: list[str]
    initial_prompt: str


class TranslatorConfig(BaseModel):
    language: str
    gemini_model_id: str


class CondenserConfig(BaseModel):
    padding_ms: int


class SubProcessorConfig(BaseModel):
    spacy_model: str


class Config(BaseModel):
    transcriber: TranscriberConfig
    translator: TranslatorConfig
    condenser: CondenserConfig
    sub_processor: SubProcessorConfig


def deep_merge(
    base: MutableMapping[str, Any], update: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merges the 'update' dictionary into the 'base' dictionary."""
    for key, value in update.items():
        if (
            isinstance(value, MutableMapping)
            and key in base
            and isinstance(base[key], MutableMapping)
        ):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    default_path: Path = Path('config.default.yaml'),
    local_path: Path = Path('config.local.yaml'),
) -> Config:
    """Loads a default config and merges a local config over it."""
    with open(default_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        with open(local_path, 'r') as file:
            local_config = yaml.safe_load(file)
            config = deep_merge(config, local_config)
    except FileNotFoundError:
        print('No local config found. Using default settings')

    return Config(**config)


SETTINGS = load_config()
