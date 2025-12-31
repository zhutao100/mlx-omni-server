# Responses Component Analysis

The `responses` component is an **adapter layer** on top of the existing `chat` component. It re-uses `chat_generation_service` for all model execution, while translating wire formats and providing a small amount of Responses-specific state handling (IDs, background jobs, event replay).

## API and Schema

-   **Endpoints:** Exposes both `/responses` and `/v1/responses` variants.
    -   `POST /v1/responses` (create; stream or non-stream)
    -   `GET /v1/responses/{response_id}` (retrieve latest response object)
    -   `GET /v1/responses/{response_id}?stream=true` (SSE replay/progress from the in-memory event log)
    -   `DELETE /v1/responses/{response_id}` (delete in-memory record)
    -   `POST /v1/responses/{response_id}/cancel` (best-effort cancel; primarily for background responses)
    -   `GET /v1/responses/{response_id}/input_items` (inspect resolved input items with basic pagination)
-   **Schema:** Modeled after the OpenAI Responses API (`ResponseRequest`, `ResponseResponse`, `ResponseStreamEvent`). Streaming uses SSE event names like `response.created`, `response.output_text.delta`, and `response.completed` (no `[DONE]` sentinel).

## Core Logic (`router.py` and `adapter.py`)

-   **Adapter Pattern:** The component's primary function is to adapt the `Response` format to the server's internal `ChatCompletion` format.
    -   The `router.py` file receives a `ResponseRequest`, immediately converts it to a `ChatCompletionRequest` using a function from `adapter.py`.
    -   It then delegates the actual model generation to the existing `chat_generation_service`, thereby re-using all of its logic, including caching, stream multiplexing, and the shared MLX gate in the inference runtime.
    -   After receiving the result from the chat service, it uses functions and classes from `adapter.py` to convert the `ChatCompletionResponse` (or stream of `ChatCompletionChunk`s) back into the `ResponseResponse` format (or stream of `ResponseStreamEvent`s).
-   **No Direct MLX Interaction:** This component does not have its own service class and does not interact with any MLX libraries directly. It is purely a data transformation layer.
-   **Structured output compatibility:** Responses-style `text.format` is mapped to chat `response_format` so existing JSON-schema enforcement can be reused.
-   **Chaining:** `previous_response_id` is supported by storing a compact “history messages” list per response and prepending it to the next request.
-   **Streaming lifecycle:** `ResponseStreamAdapter` emits a Responses-style SSE event stream including `response.in_progress`, and adds `response.content_part.done` so clients can close content parts cleanly.

## State handling (`registry.py`)

To support retrieval, cancellation, and background execution, the component maintains a small in-memory registry:

-   Stores the latest Response object per `response_id`.
-   Stores the original resolved input messages (for `/input_items`).
-   Stores streaming events for SSE replay (`GET ...?stream=true`).
-   Stores derived “history messages” for `previous_response_id` chaining.

Records are TTL-based and are not persisted across server restarts.

## Summary

The `responses` component provides an OpenAI-compatible Responses API facade over the server’s chat engine. It combines format translation (request/response + SSE events) with lightweight in-memory state to support additional endpoints (retrieve/delete/cancel/input_items), background mode, and response-chaining via `previous_response_id`.
