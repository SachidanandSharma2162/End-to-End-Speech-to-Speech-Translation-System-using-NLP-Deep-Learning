"""
app.py — Gradio Web UI
Member 7: UI & Integration Engineer

Run with:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import SpeechTranslationPipeline
from src.translation import LANGUAGE_NAMES, MODEL_MAP

# -----------------------------------------------------------------------
# Initialise pipeline (loaded once at startup)
# -----------------------------------------------------------------------
pipeline = SpeechTranslationPipeline(whisper_model="base", output_dir="outputs")

# Build language choices for dropdowns
LANG_CHOICES = [(f"{name} ({code})", code) for code, name in LANGUAGE_NAMES.items()]


# -----------------------------------------------------------------------
# Handler functions (called by Gradio)
# -----------------------------------------------------------------------

def translate_audio(audio_path, src_lang_sel, tgt_lang_sel):
    """Called when user submits an audio recording."""
    if audio_path is None:
        return "⚠️ Please record or upload an audio file.", "", None

    src_code = src_lang_sel if src_lang_sel != "auto" else None
    tgt_code = tgt_lang_sel

    try:
        result = pipeline.run(audio_path, src_lang=src_code, tgt_lang=tgt_code)
        transcript = f"**Original ({result['src_lang']}):**\n{result['clean_transcript']}"
        translation = f"**Translation ({result['tgt_lang']}):**\n{result['translated_text']}"
        return transcript, translation, result["audio_path"]
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None


def translate_text_input(text, src_lang_sel, tgt_lang_sel):
    """Called when user submits plain text (no audio)."""
    if not text.strip():
        return "⚠️ Please enter some text.", "", None

    src_code = src_lang_sel
    tgt_code = tgt_lang_sel

    try:
        result = pipeline.translate_text_only(text, src_lang=src_code, tgt_lang=tgt_code)
        transcript = f"**Input ({src_code}):**\n{result['clean_transcript']}"
        translation = f"**Translation ({tgt_code}):**\n{result['translated_text']}"
        return transcript, translation, result["audio_path"]
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None


def get_supported_pairs():
    pairs = [f"{s} → {t}" for (s, t) in MODEL_MAP.keys()]
    return "\n".join(pairs)


# -----------------------------------------------------------------------
# Gradio UI layout
# -----------------------------------------------------------------------

with gr.Blocks(title="Multilingual Speech Translator", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌐 Intelligent Multilingual Speech Translator
        Speak or type in one language — get translated text **and** synthesized speech back.
        
        **Powered by:** OpenAI Whisper (ASR) · Helsinki-NLP MarianMT (Translation) · gTTS (TTS)
        """
    )

    with gr.Tabs():

        # ---- Tab 1: Audio Input ----------------------------------------
        with gr.Tab("🎙️ Speech Input"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record or upload audio",
                    )
                    src_lang_audio = gr.Dropdown(
                        choices=[("Auto-detect", "auto")] + LANG_CHOICES,
                        value="auto",
                        label="Source language",
                    )
                    tgt_lang_audio = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="fr",
                        label="Target language",
                    )
                    btn_audio = gr.Button("🔄 Translate", variant="primary")

                with gr.Column():
                    out_transcript_audio = gr.Markdown(label="Transcript")
                    out_translation_audio = gr.Markdown(label="Translation")
                    out_audio = gr.Audio(label="🔊 Synthesized Translation", type="filepath")

            btn_audio.click(
                fn=translate_audio,
                inputs=[audio_input, src_lang_audio, tgt_lang_audio],
                outputs=[out_transcript_audio, out_translation_audio, out_audio],
            )

        # ---- Tab 2: Text Input ----------------------------------------
        with gr.Tab("⌨️ Text Input"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        lines=4,
                        placeholder="Type your text here...",
                        label="Input text",
                    )
                    src_lang_text = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="en",
                        label="Source language",
                    )
                    tgt_lang_text = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="fr",
                        label="Target language",
                    )
                    btn_text = gr.Button("🔄 Translate", variant="primary")

                with gr.Column():
                    out_transcript_text = gr.Markdown(label="Input (cleaned)")
                    out_translation_text = gr.Markdown(label="Translation")
                    out_audio_text = gr.Audio(label="🔊 Synthesized Translation", type="filepath")

            btn_text.click(
                fn=translate_text_input,
                inputs=[text_input, src_lang_text, tgt_lang_text],
                outputs=[out_transcript_text, out_translation_text, out_audio_text],
            )

        # ---- Tab 3: Supported Language Pairs --------------------------
        with gr.Tab("ℹ️ Supported Languages"):
            gr.Markdown("### Supported translation pairs")
            gr.Textbox(
                value=get_supported_pairs(),
                label="Language pairs",
                lines=15,
                interactive=False,
            )

    gr.Markdown(
        """
        ---
        **Team project** · NLP Course · 2026  
        Members: ASR (1,2) · Preprocessing (3) · Translation (4,5) · TTS (6) · UI (7) · Lead (8)
        """
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)