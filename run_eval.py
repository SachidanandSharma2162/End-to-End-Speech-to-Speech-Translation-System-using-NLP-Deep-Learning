from src.pipeline import SpeechTranslationPipeline
from src.evaluation import compute_bleu, compute_corpus_bleu, evaluate_pipeline

pipeline = SpeechTranslationPipeline(whisper_model="base")

# Your test sentences and their human reference translations
test_cases = [
    {
        "src_lang": "en", "tgt_lang": "fr",
        "source":    "The weather is really nice today.",
        "reference": "Il fait vraiment beau aujourd'hui."
    },
    {
        "src_lang": "en", "tgt_lang": "fr",
        "source":    "I would like a glass of water please.",
        "reference": "Je voudrais un verre d'eau s'il vous plaît."
    },
    {
        "src_lang": "en", "tgt_lang": "de",
        "source":    "Where is the nearest train station?",
        "reference": "Wo ist der nächste Bahnhof?"
    },
]

# Run translation and collect hypotheses
for case in test_cases:
    result = pipeline.translate_text_only(
        case["source"], case["src_lang"], case["tgt_lang"]
    )
    case["hypothesis"] = result["translated_text"]

# Print full evaluation report
evaluate_pipeline(test_cases)

# Also compute corpus-level BLEU for EN→FR only
fr_cases = [c for c in test_cases if c["tgt_lang"] == "fr"]
corpus = compute_corpus_bleu(
    hypotheses=[c["hypothesis"] for c in fr_cases],
    references=[[c["reference"]] for c in fr_cases]
)
print(f"\nCorpus BLEU (EN→FR): {corpus['bleu']}")