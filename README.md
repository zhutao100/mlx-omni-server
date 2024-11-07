# MLX Omni Server

MLX Omni Server is a local inference server powered by Apple's MLX framework, specifically designed for Apple Silicon (M-series) chips. It implements
OpenAI-compatible API endpoints, enabling seamless integration with existing OpenAI SDK clients while leveraging the power of local ML inference.

## Features

- 🚀 **Apple Silicon Optimized**: Built on MLX framework, optimized for M1/M2/M3/M4 series chips
- 🔌 **OpenAI API Compatible**: Drop-in replacement for OpenAI API endpoints
- 🎯 **Multiple AI Capabilities**:
    - Audio Processing:
        - Text-to-Speech (TTS)
        - Speech-to-Text (STT/ASR)
    - Chat Completion (Coming Soon)
    - Image Generation (Coming Soon)
- ⚡ **High Performance**: Local inference with hardware acceleration
- 🔐 **Privacy-First**: All processing happens locally on your machine
- 🛠 **SDK Support**: Works with official OpenAI SDK and other compatible clients

## Installation

```bash
# Install using pip
pip install mlx-omni-server

# Or install using poetry
poetry add mlx-omni-server
```

## Quick Start

1. Start the server:

```bash
# If installed via pip as a package
mlx-omni-server start

# If installed via poetry (recommended during development)
poetry run start
```

2. Use with OpenAI SDK:

```python
from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Point to local server
    api_key="not-needed"  # API key is not required for local server
)

# Text-to-Speech Example
response = client.audio.speech.create(
    model="lucasnewman/f5-tts-mlx",
    input="Hello, welcome to MLX Omni Server!"
)

# Speech-to-Text Example
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="mlx-community/whisper-large-v3-turbo",
    file=audio_file
)

```

## API Endpoints

The server implements OpenAI-compatible endpoints:

- `/v1/audio/speech` - Text-to-Speech
- `/v1/audio/transcriptions` - Speech-to-Text
- `/v1/chat/completions` - Chat completions (Coming Soon)
- `/v1/images/generations` - Image generation (Coming Soon)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to
change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MLX](https://github.com/ml-explore/mlx) by Apple
- API design inspired by [OpenAI](https://openai.com)
- Uses [FastAPI](https://fastapi.tiangolo.com/) for the server implementation
- Text-to-Speech powered by [lucasnewman/f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx)
- Speech-to-Text powered by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

## Disclaimer

This project is not affiliated with or endorsed by OpenAI or Apple. It's an independent implementation that provides OpenAI-compatible APIs using
Apple's MLX framework.