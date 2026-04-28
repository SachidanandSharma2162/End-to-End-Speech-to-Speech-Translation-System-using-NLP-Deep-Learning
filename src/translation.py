"""
translation.py — Neural Machine Translation using HuggingFace MarianMT
Members 4 & 5: Translation Model Engineers
"""

from transformers import MarianMTModel, MarianTokenizer
import torch


# Language pair → HuggingFace Helsinki-NLP model name
MODEL_MAP = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
}

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German",
    "hi": "Hindi",   "es": "Spanish", "zh": "Chinese",
    "ar": "Arabic",  "ru": "Russian",
}


class Translator:
    """
    Loads a Helsinki-NLP MarianMT model for a given language pair
    and translates text. Models are cached after the first load.
    """

    def __init__(self):
        self._cache: dict = {}   # (src, tgt) → (tokenizer, model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supported_pairs(self) -> list[tuple[str, str]]:
        return list(MODEL_MAP.keys())

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from src_lang to tgt_lang.

        Args:
            text:     Input text (pre-cleaned by TextPreprocessor)
            src_lang: ISO 639-1 source language code  e.g. 'en'
            tgt_lang: ISO 639-1 target language code  e.g. 'fr'

        Returns:
            Translated string.

        Raises:
            ValueError if the language pair is not supported.
        """
        if src_lang == tgt_lang:
            return text

        pair = (src_lang, tgt_lang)
        if pair not in MODEL_MAP:
            raise ValueError(
                f"Language pair {src_lang}→{tgt_lang} is not supported.\n"
                f"Supported pairs: {list(MODEL_MAP.keys())}"
            )

        tokenizer, model = self._load(pair)

        # Chunk long texts into sentences to avoid token limit (512)
        chunks = self._chunk(text)
        translated_chunks = []

        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, num_beams=4, early_stopping=True)

            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            translated_chunks.append(decoded)

        return " ".join(translated_chunks)

    def translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate a list of strings in one call (more efficient for many sentences)."""
        return [self.translate(t, src_lang, tgt_lang) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, pair: tuple[str, str]):
        if pair not in self._cache:
            model_name = MODEL_MAP[pair]
            print(f"[Translation] Loading model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self._cache[pair] = (tokenizer, model)
            print(f"[Translation] Model ready on {self.device}.")
        return self._cache[pair]

    def _chunk(self, text: str, max_words: int = 100) -> list[str]:
        """Split long text into manageable sentence-sized chunks."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], []
        word_count = 0

        for sentence in sentences:
            words = len(sentence.split())
            if word_count + words > max_words and current:
                chunks.append(" ".join(current))
                current, word_count = [], 0
            current.append(sentence)
            word_count += words

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]