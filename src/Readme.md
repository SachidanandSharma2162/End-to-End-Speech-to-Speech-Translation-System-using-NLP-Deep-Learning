# 🎙️ Speech-to-Speech Translation Pipeline

A modular end-to-end Speech-to-Speech Translation System that converts input speech into translated speech using ASR, preprocessing, translation, and TTS.

## 🚀 Features
- Automatic Speech Recognition (ASR)
- Text Preprocessing
- Language Translation
- Text-to-Speech (TTS)
- BLEU Score Evaluation
- Gradio Web UI

## 📁 Project Structure
.
├── requirements.txt
├── asr.py
├── preprocessing.py
├── translation.py
├── tts.py
├── evaluation.py
├── pipeline.py
└── app.py

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>
python -m venv venv
```

Activate:
- Windows: venv\Scripts\activate
- Mac/Linux: source venv/bin/activate

```bash
pip install -r requirements.txt
```

## ▶️ Run Project

### Run UI
```bash
python app.py
```
Open: http://127.0.0.1:7860

### Run Pipeline
```bash
python pipeline.py
```

## 🔄 Workflow
Speech → ASR → Preprocessing → Translation → TTS → Output + BLEU Score

 
