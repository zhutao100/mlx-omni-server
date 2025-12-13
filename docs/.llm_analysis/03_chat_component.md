# Chat Component Analysis

The `chat` component is the most complex part of the `mlx-omni-server`, providing OpenAI-compatible chat completion functionality with advanced features.

## API and Schema

-   **Endpoints:** Exposes `/chat/completions` and `/v1/chat/completions`.
-   **Schema:** The Pydantic models in `schema.py` closely mirror the OpenAI API, ensuring drop-in compatibility for many clients.
-   **Core Features defined in Schema:**
    -   **Multimodality:** Can process text, images (`image_url`), and audio (`input_audio`) within the same chat conversation.
    -   **Tool Use / Function Calling:** Comprehensive support for defining tools, passing them to the model, and parsing the model's `tool_calls` output.
    -   **Structured Output:** Supports `json_object` mode and a more powerful `json_schema` mode to force the model's output to conform to a specific structure.
    -   **Streaming:** Full support for streaming responses via Server-Sent Events (SSE).

## Core Logic (`generation_service.py`)

The `ChatGenerationService` orchestrates the entire process.

-   **Concurrency Model:**
    -   A global `mlx_lock` ensures that only **one MLX model generation runs at a time**. This serializes all heavy computation to avoid GPU contention.
    -   Blocking model generation code is run in a separate thread pool (`run_in_threadpool`) to keep the main async event loop responsive.

-   **Caching and Stream Multiplexing:**
    -   A sophisticated in-memory caching system is implemented to handle identical requests.
    -   For streaming requests, multiple clients can connect to the same ongoing generation. The server sends the already-generated chunks (replay) and then streams the new ones live to all clients.
    -   The system is resource-aware, automatically cancelling background generation tasks if all clients disconnect.
    -   A background task periodically cleans up cache entries older than 5 minutes.

-   **Dynamic Model Loading:**
    -   Models are loaded dynamically based on the `model` field in the request payload.
    -   The system supports advanced features like loading LoRA adapters and "draft models" for speculative decoding, which are passed as non-standard parameters in the request.

## Supporting Modules

-   **`router.py`:** A clean FastAPI router that handles HTTP requests and delegates all logic to the `chat_generation_service`.
-   **`mlx_lm/` & `mlx_vlm/`:** These contain the abstraction layers that wrap the `mlx-lm` and `mlx-vlm` libraries, providing a consistent interface for text and multimodal models. They likely contain the implementation for prompt caching and logits processing (for JSON mode).
-   **`tools/`:** Contains model-specific logic for handling tool calls. This is where the generic OpenAI tool format is translated into the specific format required by models like Llama 3, Mistral, etc.
-   **`templates/`:** Jinja templates are used to apply the correct chat template for different models, ensuring the prompt is formatted correctly.

In summary, the chat component is a production-quality system with a strong focus on performance (caching, stream multiplexing), compatibility (OpenAI-like API), and advanced features (multimodality, tool use, structured output). The serialization of MLX tasks is a key architectural decision that has significant performance implications.
