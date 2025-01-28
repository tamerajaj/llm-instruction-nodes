import os
from typing import Dict, Any, Tuple

import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.app.evaluate_llm import compute_metrics
from src.app.prompts import prompt


def load_model(
    model_name: str, model_dir: str = "/model_storage"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and return a model and tokenizer from Hugging Face, saving them to a persistent directory.

    Args:
        model_name (str): The name of the Hugging Face model to load.
        model_dir (str): The directory where the model will be cached. Defaults to "/model_storage".

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Create a directory (if it does not exist) based on the model name
    model_path = os.path.join(model_dir, model_name.replace("/", "_"))
    os.makedirs(model_path, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # Leverages GPUs if available
        cache_dir=model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

    return model, tokenizer


def llm_inference(
    user_request: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
) -> str:
    """
    Generate text from the language model based on the given user input.

    Args:
        user_request (str): The user's prompt or query.
        model (AutoModelForCausalLM): An instance of a language model.
        tokenizer (AutoTokenizer): A tokenizer corresponding to the model.

    Returns:
        str: The generated text from the model.
    """
    # Construct the input text
    input_text = f"{prompt}\nPrompt:\n {user_request}\n Sequence of Nodes:\n"

    # Initialize the text-generation pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # Generate text
    generation = pipe(
        input_text, max_new_tokens=50, num_return_sequences=1, return_full_text=False
    )[0]["generated_text"]

    # Remove the initial prompt if it's repeated
    if input_text in generation:
        generation = generation.replace(input_text, "").strip()

    return generation.strip()


def main() -> None:
    """
    Main function that loads the model, performs inference, and prints metrics.
    """
    # Change the model name as needed
    model_name = "google/gemma-2-2b-it"

    # Example user request
    user_request = "Highlight an element when the mouse enters it and remove the highlight when the mouse leaves."

    # Expected output for evaluation
    expected_output = """
    1. [OnMouseEnter],
    2. [Highlight],
    3. [OnMouseLeave]
    """

    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)

    # Perform inference
    response = llm_inference(user_request, model, tokenizer)
    print("Response:", response)

    # Compute and print metrics
    metrics: Dict[str, Any] = compute_metrics(response, expected_output)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
