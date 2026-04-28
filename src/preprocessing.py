"""
preprocessing.py — Text Preprocessing after ASR
Member 3: NLP / Text Preprocessing Engineer
"""

import re
import unicodedata


class TextPreprocessor:
    """
    Cleans and normalises raw ASR output before it enters the translation model.
    Good preprocessing directly improves BLEU scores.
    """

    # Common ASR artifacts to strip
    FILLER_PATTERNS = [
        r"\bum+\b", r"\buh+\b", r"\bhmm+\b", r"\blike\b(?=\s+\b)",
        r"\byou know\b", r"\bi mean\b",
    ]

    def __init__(self, aggressive: bool = False):
        """
        Args:
            aggressive: If True, also remove filler words common in spoken language.
        """
        self.aggressive = aggressive
        self._filler_re = re.compile(
            "|".join(self.FILLER_PATTERNS), flags=re.IGNORECASE
        ) if aggressive else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """
        Full preprocessing pipeline. Returns cleaned text ready for translation.
        Steps:
          1. Unicode normalisation (NFC)
          2. Strip leading/trailing whitespace
          3. Collapse multiple spaces/newlines
          4. Fix common punctuation issues from ASR (missing periods, etc.)
          5. (Optional) Remove filler words
        """
        text = self._normalize_unicode(text)
        text = text.strip()
        text = self._collapse_whitespace(text)
        text = self._fix_punctuation(text)
        if self.aggressive and self._filler_re:
            text = self._remove_fillers(text)
        text = self._capitalize_sentences(text)
        return text

    def segment_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for chunk-based translation.
        Translating sentence-by-sentence gives better results on long inputs.
        """
        # Simple rule-based splitter (works well for clean ASR output)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _collapse_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)

    def _fix_punctuation(self, text: str) -> str:
        # Remove spaces before punctuation
        text = re.sub(r"\s([?.!,;:])", r"\1", text)
        # Ensure sentence ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _remove_fillers(self, text: str) -> str:
        cleaned = self._filler_re.sub("", text)
        return self._collapse_whitespace(cleaned)

    def _capitalize_sentences(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(s.capitalize() for s in sentences if s)