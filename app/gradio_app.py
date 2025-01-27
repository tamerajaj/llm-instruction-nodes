import gradio as gr

from evaluate_llm import compute_metrics
from llm import load_model, llm_inference


model_name = "google/gemma-2-2b-it"
model, tokenizer = load_model(model_name)



def generate_response(input_text: str) -> str:
    """
    Generates a response using the preloaded model and tokenizer.

    Args:
        input_text (str): The user input text to be processed by the model.

    Returns:
        str: The generated model output.
    """
    return llm_inference(input_text, model, tokenizer)



def evaluate_response(predicted_output: str, expected_output: str) -> str:
    """
    Evaluates the predicted output against the expected output and returns metrics.

    Args:
        predicted_output (str): The model-generated text to be evaluated.
        expected_output (str): The user-defined reference text.

    Returns:
        str: Formatted string containing BLEU, ROUGE, and MED (Average) scores.
    """
    try:
        metrics = compute_metrics(predicted_output, expected_output)
        return (
            f"Metrics:\n"
            f"BLEU: {metrics['BLEU']}\n"
            f"ROUGE: {metrics['ROUGE']}\n"
            f"MED (Average): {metrics['MED (Average)']}"
        )
    except Exception as e:
        return f"Error in evaluation: {str(e)}"



def clear_fields() -> tuple[str, str, str]:
    """
    Clears all text fields in the interface.

    Returns:
        tuple[str, str, str]: Empty strings for the input, prediction, and expected output fields.
    """
    return "", "", ""



with gr.Blocks() as demo:
    gr.Markdown("# LLM Interface with Evaluation")
    gr.Markdown(
        "Use the text boxes below to generate a response from the LLM and then "
        "optionally evaluate the response against an expected output."
    )

    # Textboxes
    input_text = gr.Textbox(
        label="Input Text", lines=5, placeholder="Enter your prompt here..."
    )
    predicted_output = gr.Textbox(
        label="Predicted Output",
        lines=5,
        placeholder="Generated response will appear here...",
    )
    expected_output = gr.Textbox(
        label="Expected Output",
        lines=5,
        placeholder="Enter your reference or expected output here...",
    )
    evaluation_metrics = gr.Textbox(
        label="Evaluation Metrics",
        lines=5,
        placeholder="Evaluation results will appear here...",
        interactive=False,
    )

    # Buttons
    submit_button = gr.Button("Generate Response")
    evaluate_button = gr.Button("Evaluate")
    clear_button = gr.Button("Clear")

    # Button Click Handlers
    submit_button.click(
        fn=generate_response, inputs=input_text, outputs=predicted_output
    )
    evaluate_button.click(
        fn=evaluate_response,
        inputs=[predicted_output, expected_output],
        outputs=evaluation_metrics,
    )
    clear_button.click(
        fn=clear_fields,
        inputs=None,
        outputs=[input_text, predicted_output, expected_output],
    )

if __name__ == "__main__":
    demo.launch()
