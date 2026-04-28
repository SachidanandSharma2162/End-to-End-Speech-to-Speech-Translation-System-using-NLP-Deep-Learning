"""
tts.py — Text-to-Speech using gTTS
Member 6: TTS & Output Engineer
"""

from gtts import gTTS
from gtts.lang import tts_langs
import os
import tempfile


# Map ISO 639-1 codes to gTTS-compatible language codes
# gTTS uses the same codes for most languages
GTTS_LANG_MAP = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "es": "es",
    "zh": "zh-CN",
    "ar": "ar",
    "ru": "ru",
}


class TTSEngine:
    """
    Converts translated text to spoken audio using Google Text-to-Speech (gTTS).
    Saves output as an MP3 file and returns the path.
    """

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save audio files.
                        Defaults to system temp directory.
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, language: str, filename: str = None) -> str:
        """
        Convert text to speech and save as an MP3 file.

        Args:
            text:     Text to synthesize (translated output).
            language: ISO 639-1 language code (e.g. 'fr', 'hi').
            filename: Optional output filename (without extension).
                      Auto-generated if not provided.

        Returns:
            Absolute path to the saved MP3 file.

        Raises:
            ValueError if the language is not supported by gTTS.
        """
        gtts_lang = self._resolve_lang(language)

        if not filename:
            import uuid
            filename = f"tts_{language}_{uuid.uuid4().hex[:8]}"

        output_path = os.path.join(self.output_dir, f"{filename}.mp3")

        print(f"[TTS] Synthesizing {language} speech → {output_path}")
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(output_path)
        print("[TTS] Audio saved.")

        return output_path

    def supported_languages(self) -> dict:
        """Returns dict of supported language code → language name."""
        return {k: v for k, v in GTTS_LANG_MAP.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_lang(self, language: str) -> str:
        if language not in GTTS_LANG_MAP:
            raise ValueError(
                f"TTS language '{language}' is not supported.\n"
                f"Supported: {list(GTTS_LANG_MAP.keys())}"
            )
        return GTTS_LANG_MAP[language]