from src.app.evaluate_llm import compute_metrics
from src.app.llm import load_model

def test_compute_metrics_basic():
    predictions = "Line 1\nLine 2\nLine 3\nLine 4"
    references = "Line 1\nLine 2\nLine 3\nLine 4"

    metrics = compute_metrics(predictions, references)
    assert "BLEU" in metrics
    assert "ROUGE" in metrics
    assert "MED (Average)" in metrics
    # Since they're identical lines, BLEU/ROUGE should be high, MED low.
    print(metrics)
    assert metrics["MED (Average)"] == 0
    # TODO: add assert for rouge and bleu

def test_load_model_no_crash():
    model_name = "google/gemma-2-2b-it"
    model, tokenizer = load_model(model_name)
    assert model is not None
    assert tokenizer is not None
