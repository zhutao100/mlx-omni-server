# Development Guide

This guide is intended for developers who want to contribute to MLX Omni Server or create their own extensions.

## Setting Up Development Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zhutao100/mlx-omni-server.git
   cd mlx-omni-server
   ```

2. **Install dependencies:**

   ```bash
   # Core install (chat + responses + embeddings)
   pip install -e .

   # Optional modalities (images / STT / TTS)
   pip install -e ".[all]"
   ```

## Running the Server in Development Mode

There are two ways to run the server during development:

### 1. Using uvicorn (Recommended for development)

```bash
uvicorn mlx_omni_server.main:app --reload --host 0.0.0.0 --port 10240
```

The `--reload` flag enables hot-reload, which automatically restarts the server when code changes are detected. This is particularly useful during development.

### 2. Using the standard entry point

```bash
mlx-omni-server
```

## API Interaction Methods

MLX Omni Server supports multiple ways to interact with AI capabilities:

### 1. REST API Direct Access

```bash
# Chat completions
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# List available models
curl http://localhost:10240/v1/models
```

### 2. OpenAI SDK Integration

```python
from openai import OpenAI

# Standard client setup
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)
```

### 3. FastAPI TestClient (Development)

Perfect for testing without starting a server:

```python
from openai import OpenAI
from fastapi.testclient import TestClient
from mlx_omni_server.main import app

client = OpenAI(http_client=TestClient(app))
```

## Model Management

MLX Omni Server provides flexible model management capabilities:

### Automatic Model Downloading

When you specify a model ID that hasn\'t been downloaded yet, the framework will automatically download it from Hugging Face:

```python
response = client.chat.completions.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",  # Will download if not available
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Using Local Models

To use a locally downloaded model, simply set the `model` parameter to the local model path:

```python
response = client.chat.completions.create(
    model="/path/to/your/local/model",  # Local model path
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Listing Available Models

You can check which models are available on your system:

```bash
curl http://localhost:10240/v1/models
```

Or using the Python client:

```python
response = client.models.list()
for model in response.data:
    print(f"Model ID: {model.id}")
```

## Performance Optimization

### Model Selection

| Quantization | Benefits | Best For |
|--------------|----------|----------|
| **4-bit** | Fast inference, low memory | Everyday use, M1/M2 Macs |
| **8-bit** | Better quality, still fast | High-quality results |
| **DWQ** | Optimized for MLX | Specialized workloads |

### Hardware Recommendations

| Component | Recommendation |
|-----------|----------------|
| **Memory** | 16GB+ RAM for larger models |
| **Storage** | SSD for faster loading |
| **Cooling** | Adequate for sustained performance |

### Production Configuration

```bash
# Multi-worker setup for better throughput
mlx-omni-server --workers 2 --log-level warning --log-file
```

## Troubleshooting

Common issues and their solutions:

### Server startup troubleshooting

| Issue | Solution |
|-------|----------|
| **Apple Silicon required** | Ensure you have M1/M2/M3/M4 Mac |
| **Python version** | Use Python 3.11+ |
| **Installation** | Verify latest version installed |
| **Debug info** | Check logs for detailed errors |

### Model Download Issues

If you encounter problems downloading models:

1. Check your internet connection
2. Verify you have sufficient disk space
3. Try downloading the model directly with Hugging Face tools:

```bash
huggingface-cli download mlx-community/gemma-3-1b-it-4bit-DWQ
```

### Memory Errors

If you get out-of-memory errors:

1. Use a smaller quantized model (4bit instead of 8bit)
2. Close other memory-intensive applications
3. Restart the server with a fresh process

### Performance Issues

If responses are slow:

1. Ensure you\'re using quantized models
2. Check that you have adequate cooling (thermal throttling can reduce performance)
3. Consider using a model better suited to your hardware


## Contributing Guidelines

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Follow the code style:
   - Use [Black](https://black.readthedocs.io/) for Python code formatting
   - Use [isort](https://pycqa.github.io/isort/) for import sorting
   - Run pre-commit hooks before committing:
     ```bash
     pre-commit install
     pre-commit run --all-files
     ```
4. Write clear commit messages
5. Push to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
6. Open a Pull Request with:
   - Clear description of the changes
   - Any relevant issue numbers
   - Screenshots for UI changes (if applicable)

## Testing

Run the test suite:
```bash
pytest
```

## Building Documentation

The documentation is written in Markdown and stored in the `docs/` directory.

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in the GitHub Discussions section
- Check existing issues and pull requests before creating new ones
