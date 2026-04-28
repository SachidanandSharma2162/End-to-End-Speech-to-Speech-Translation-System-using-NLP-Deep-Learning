"""
pipeline.py — Full Translation Pipeline Orchestrator
Wires ASR → Preprocessing → Translation → TTS together.
"""

import os
from src.asr import ASREngine
from src.preprocessing import TextPreprocessor
from src.translation import Translator, LANGUAGE_NAMES
from src.tts import TTSEngine


class SpeechTranslationPipeline:
    """
    End-to-end pipeline:
      Audio file → Whisper ASR → Text cleaning → MarianMT translation → gTTS audio

    Usage:
        pipeline = SpeechTranslationPipeline()
        result = pipeline.run("input.wav", src_lang="en", tgt_lang="fr")
        print(result["translated_text"])
        # result["audio_path"] → path to synthesized MP3
    """

    def __init__(
        self,
        whisper_model: str = "base",
        output_dir: str = "outputs",
        aggressive_clean: bool = False,
    ):
        print("[Pipeline] Initialising components...")
        self.asr = ASREngine(model_size=whisper_model)
        self.preprocessor = TextPreprocessor(aggressive=aggressive_clean)
        self.translator = Translator()
        self.tts = TTSEngine(output_dir=output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("[Pipeline] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        audio_path: str,
        src_lang: str = None,
        tgt_lang: str = "en",
    ) -> dict:
        """
        Run the full pipeline on an audio file.

        Args:
            audio_path: Path to input audio file (.wav / .mp3 / .m4a).
            src_lang:   Source spoken language code. None = auto-detect.
            tgt_lang:   Target language code for translation + TTS.

        Returns:
            dict:
                'raw_transcript'    — Whisper's raw output
                'clean_transcript'  — After preprocessing
                'translated_text'   — MarianMT translation
                'audio_path'        — Path to synthesized MP3
                'src_lang'          — Detected / specified source lang
                'tgt_lang'          — Target language
        """
        # Step 1: ASR
        print("\n[Pipeline] Step 1/4 — Transcribing speech...")
        asr_result = self.asr.transcribe(audio_path, language=src_lang)
        raw_text = asr_result["text"]
        detected_src = asr_result["language"]
        print(f"  Transcript ({detected_src}): {raw_text}")

        # Step 2: Preprocessing
        print("[Pipeline] Step 2/4 — Cleaning text...")
        clean_text = self.preprocessor.clean(raw_text)
        print(f"  Cleaned: {clean_text}")

        # Step 3: Translation
        print(f"[Pipeline] Step 3/4 — Translating {detected_src} → {tgt_lang}...")
        translated = self.translator.translate(clean_text, detected_src, tgt_lang)
        print(f"  Translation: {translated}")

        # Step 4: TTS
        print("[Pipeline] Step 4/4 — Synthesizing speech...")
        audio_out = self.tts.synthesize(translated, tgt_lang)

        print("\n[Pipeline] ✓ Done.")
        return {
            "raw_transcript": raw_text,
            "clean_transcript": clean_text,
            "translated_text": translated,
            "audio_path": audio_out,
            "src_lang": detected_src,
            "tgt_lang": tgt_lang,
        }

    def translate_text_only(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> dict:
        """
        Translate plain text (no audio input). Useful for testing the
        preprocessing + translation stages without Whisper.
        """
        clean = self.preprocessor.clean(text)
        translated = self.translator.translate(clean, src_lang, tgt_lang)
        audio_out = self.tts.synthesize(translated, tgt_lang)
        return {
            "clean_transcript": clean,
            "translated_text": translated,
            "audio_path": audio_out,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }