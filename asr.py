"""
asr.py — Automatic Speech Recognition using OpenAI Whisper
Members 1 & 2: ASR Engineers
"""

import whisper
import numpy as np
import torch


class ASREngine:
    """
    Wraps OpenAI Whisper for speech-to-text transcription.
    Supports audio file input and automatic language detection.
    """

    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, model_size: str = "base"):
        if model_size not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_size must be one of {self.SUPPORTED_MODELS}")
        print(f"[ASR] Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size)
        self.model_size = model_size
        print("[ASR] Model loaded.")

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to .wav / .mp3 / .m4a audio file
            language:   ISO 639-1 code (e.g. 'en', 'hi', 'fr').
                        Pass None for automatic detection.

        Returns:
            dict with keys:
                'text'     — raw transcription string
                'language' — detected/specified language code
                'segments' — list of timed segment dicts
        """
        options = {}
        if language:
            options["language"] = language

        print(f"[ASR] Transcribing: {audio_path}")
        result = self.model.transcribe(audio_path, **options)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", language or "unknown"),
            "segments": result.get("segments", []),
        }

    def detect_language(self, audio_path: str) -> str:
        """
        Detect the spoken language without full transcription (faster).

        Returns: ISO 639-1 language code string
        """
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected = max(probs, key=probs.get)
        print(f"[ASR] Detected language: {detected} (confidence: {probs[detected]:.2f})")
        return detected