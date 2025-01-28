import pytest

from src.app.evaluate_llm import compute_metrics
from src.app.llm import load_model, llm_inference


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "google/gemma-2-2b-it"
    model, tokenizer = load_model(model_name)

    return model, tokenizer

def test_end_to_end_inference(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Navigate to a new page after a delay of 3 seconds when the user clicks a button."
    predicted_output = llm_inference(prompt, model, tokenizer)

    # Check if output is non-empty and contains bracketed nodes
    assert len(predicted_output.strip()) > 0
    assert "[" in predicted_output and "]" in predicted_output

def test_end_to_end_evaluation(model_and_tokenizer):

    prompt = "Navigate to a new page after a delay of 3 seconds when the user clicks a button."
    expected_output = """
    1. [OnClick]
    2. [Delay]
    3. [Navigate]
    """
    model, tokenizer = model_and_tokenizer
    predicted_output = llm_inference(prompt, model, tokenizer)
    metrics = compute_metrics(predicted_output, expected_output)

    assert "BLEU" in metrics
    assert "ROUGE" in metrics
    assert "MED (Average)" in metrics
