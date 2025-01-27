
# LLM Demo for instructions -> nodes mapping, with Gradio 

A demonstration of running a Large Language Model (LLM) using PyTorch, Gradio, and Hugging Face libraries. This setup includes:

- **Model Loading & Inference** (`llm.py`)
- **Evaluation Metrics** (`evaluate.py`)
- **Prompt Instructions** (`prompts.py`)
- **Gradio Interface** (`gradio_app.py`)
- **Dockerfile** for containerization

---

## Project Structure

```
.
├── app
│   ├── llm.py           # Model loading & inference
│   ├── evaluate.py      # BLEU, ROUGE, and MED evaluation
│   ├── gradio_app.py    # Main Gradio UI
│   └── prompts.py       # Prompt instructions & node lists
├── Dockerfile           # Docker build instructions
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (includes HF_TOKEN)
```

> **Note:** The `.env` file contains a `HF_TOKEN` variable (your Hugging Face access token) used to authenticate with private model repositories.

---

## Prerequisites

1. **Docker** (for containerized deployment).
2. **NVIDIA GPU** with drivers (and CUDA support if desired).
3. **Hugging Face Token** (if you’re using private or restricted models).

---

## Quick Start

Below are instructions to build and run this project in a Docker container.

### 1. Clone the Repository



### 2. Build the Docker Image

```bash
docker build -t gradio-llm .
```

### 3. Create or Update `.env` (Optional)

If you have a Hugging Face token (e.g., for models like `google/gemma-2-2b-it`), ensure you have a `.env` file at the root with:

```
HF_TOKEN=your_hugging_face_token
```

### 4. Run the Docker Container

Replace `\path\to\.cache\huggingface\hub` with your desired local path for caching models. Adjust `--gpus all` if you have a different GPU setup or want CPU-only.

```bash
docker run --name gradio-llm \
  -p 7860:7860 \
  --env HF_TOKEN=your_hugging_face_token \
  -v \path\to\.cache\huggingface\hub:/model_storage \
  --gpus all \
  gradio-llm
```

> **Note:**  
> - The `--env HF_TOKEN=...` can be replaced by the `.env` file if you prefer automatic loading.  
> - The `-v` flag maps your local Hugging Face cache to `/model_storage` inside the container for persistent model storage.

The Gradio app will launch on [http://localhost:7860](http://localhost:7860).

---

## Usage of Gradio App

1. **Open** your browser to [http://localhost:7860](http://localhost:7860).
2. **Enter** a text prompt in the _Input Text_ box.
3. **Click** **Generate Response** to see the LLM’s output.
4. (Optional) **Enter** an _Expected Output_ to evaluate using BLEU, ROUGE, and MED metrics.

---

## Files Overview

### `app/llm.py`
- **load_model(model_name)**: Loads a Hugging Face model (with 4-bit quantization support).
- **llm_inference(user_request, model, tokenizer)**: Generates text based on a user request using a prompt template.

### `app/evaluate.py`
- **compute_metrics(predictions, references)**: Computes BLEU, ROUGE, and Minimum Edit Distance (MED) for comparing model outputs against references.

### `app/prompts.py`
- **prompt**: A comprehensive instruction string guiding the LLM on how to respond. Contains a list of allowed nodes and examples.

### `app/gradio_app.py`
- **Gradio UI**: Defines text boxes for input, predicted output, expected output, and a metrics display box. Also includes buttons for generating responses, evaluating metrics, and clearing fields.

---

## AI Assistance tools:

ChatGPT (including variants `chatgpt-o1`, `chatgpt-4o`) and GitHub Copilot were used for debugging and suggestions.

---

## Next steps Improvements

1. Add unit tests  
2. Improve speed by using smaller models  
3. Enhance how evaluation results are displayed in Gradio  
4. Try additional evaluation approaches  
5. Refactor model name and configuration parameters