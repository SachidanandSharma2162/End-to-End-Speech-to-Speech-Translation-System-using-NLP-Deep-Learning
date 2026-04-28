"""
evaluation.py — Translation Quality Evaluation (BLEU Score)
Members 4 & 5: Translation Model Engineers
"""

import sacrebleu


def compute_bleu(hypothesis: str, references: list[str]) -> dict:
    """
    Compute BLEU score for a single translation.

    Args:
        hypothesis:  The model's translated output.
        references:  List of human reference translations.

    Returns:
        dict with keys:
            'bleu'      — BLEU score (0–100)
            'precisions'— n-gram precisions
            'brevity_penalty' — BP value
    """
    result = sacrebleu.sentence_bleu(hypothesis, references)
    return {
        "bleu": round(result.score, 2),
        "precisions": [round(p, 2) for p in result.precisions],
        "brevity_penalty": round(result.bp, 4),
    }


def compute_corpus_bleu(hypotheses: list[str], references: list[list[str]]) -> dict:
    """
    Compute corpus-level BLEU over multiple sentences.

    Args:
        hypotheses: List of translated sentences.
        references: List of reference lists. Each element is a list of
                    references for the corresponding hypothesis.

    Returns:
        dict with 'bleu' and 'precisions'.

    Example:
        hypotheses = ["Bonjour le monde", "Comment ça va"]
        references = [["Bonjour monde", "Salut le monde"], ["Comment allez-vous"]]
    """
    # sacrebleu expects references transposed: list-of-ref-lists, not list-of-sentence-lists
    transposed = list(map(list, zip(*references))) if references else [[]]
    result = sacrebleu.corpus_bleu(hypotheses, transposed)
    return {
        "bleu": round(result.score, 2),
        "precisions": [round(p, 2) for p in result.precisions],
    }


def evaluate_pipeline(pairs: list[dict]) -> None:
    """
    Pretty-print evaluation results for a list of translation pairs.

    Args:
        pairs: List of dicts, each with keys:
               'source', 'hypothesis', 'reference', 'src_lang', 'tgt_lang'
    """
    print("\n" + "=" * 60)
    print("TRANSLATION EVALUATION REPORT")
    print("=" * 60)

    for i, p in enumerate(pairs, 1):
        score = compute_bleu(p["hypothesis"], [p["reference"]])
        print(f"\n[{i}] {p['src_lang']} → {p['tgt_lang']}")
        print(f"  Source     : {p['source']}")
        print(f"  Hypothesis : {p['hypothesis']}")
        print(f"  Reference  : {p['reference']}")
        print(f"  BLEU Score : {score['bleu']}")

    print("\n" + "=" * 60)