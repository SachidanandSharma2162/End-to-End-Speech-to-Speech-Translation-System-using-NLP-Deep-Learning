🎙️ Speech-to-Speech Translation Pipeline

A modular end-to-end Speech-to-Speech Translation System that converts input speech into translated speech using multiple stages including ASR, text preprocessing, translation, and TTS, along with evaluation metrics.

🚀 Features
🎧 Automatic Speech Recognition (ASR) – Converts speech to text
🧹 Text Preprocessing – Cleans and normalizes text
🌍 Translation Module – Translates text into target language
🔊 Text-to-Speech (TTS) – Converts translated text back to speech
📊 Evaluation (BLEU Score) – Measures translation quality
💻 Gradio UI – Interactive web interface for easy usage

📁 Project Structure
.
├── requirements.txt        # Project dependencies
├── asr.py                 # ASR module (Member 1 & 2)
├── preprocessing.py       # Text preprocessing (Member 3)
├── translation.py         # Translation module (Member 4 & 5)
├── tts.py                 # Text-to-Speech (Member 6)
├── evaluation.py          # BLEU evaluation (Member 4 & 5)
├── pipeline.py            # Main pipeline orchestrator
└── app.py                 # Gradio UI (Member 7)

⚙️ Installation
1. Clone the repository
git clone <your-repo-url>
cd <your-project-folder>
2. Create virtual environment (recommended)
python -m venv venv
Activate:
Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
▶️ Usage
🔹 Run with Gradio UI (Recommended)
python app.py

Open in browser:

http://127.0.0.1:7860
🔹 Run Pipeline Directly
python pipeline.py

🔄 Pipeline Workflow
Speech Input
     ↓
ASR (Speech → Text)
     ↓
Text Preprocessing
     ↓
Translation
     ↓
Text-to-Speech (TTS)
     ↓
Translated Speech Output
     ↓
BLEU Score Evaluation

🧠 Example Flow (Pseudo Code)
def run_pipeline(audio_input):
    text = asr(audio_input)
    clean_text = preprocess(text)
    translated = translate(clean_text)
    speech = tts(translated)
    score = evaluate(translated)
    return speech, score
    
📦 Dependencies

All dependencies are listed in:

requirements.txt

Common libraries may include:

torch
transformers
gradio
nltk
ffmpeg-python
⚠️ Common Issues & Fixes
❌ ModuleNotFoundError
pip install -r requirements.txt
❌ FFmpeg not found

Install FFmpeg:

Windows: https://ffmpeg.org/download.html
Or:
pip install ffmpeg-python
❌ Slow first run
Models (e.g., HuggingFace) download on first execution.
👥 Team Contributions
Module	Responsibility
ASR	Member 1 & 2
Preprocessing	Member 3
Translation	Member 4 & 5
Evaluation	Member 4 & 5
TTS	Member 6
UI (Gradio)	Member 7
📌 Future Improvements
🌐 Support for multiple languages
⚡ Real-time streaming translation
📱 Mobile-friendly UI
🎯 Improved model accuracy
📜 License

This project is for academic and research purposes.