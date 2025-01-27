# Use the PyTorch base image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set  environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create and use the /app directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app

# Declare a volume for model storage
VOLUME ["/model_storage"]

# Expose the default Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "gradio_app.py"]
