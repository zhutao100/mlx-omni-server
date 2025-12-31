from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional
from uuid import uuid4

from ..chat.schema import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    FunctionCall,
    MultimodalContentItem,
    Role,
    ToolCall,
    ToolType,
)
from .schema import ResponseRequest, ResponseStreamEvent


@dataclass
class OutputItemState:
    key: str
    choice_index: int
    index: int
    item_id: str
    kind: Literal["message", "function_call"]
    status: str = "in_progress"
    text: str = ""
    arguments: str = ""
    function_name: Optional[str] = None
    call_id: Optional[str] = None
    done_emitted: bool = False

    def to_output_dict(self) -> Dict[str, Any]:
        if self.kind == "function_call":
            return {
                "id": self.item_id,
                "type": "function_call",
                "status": self.status,
                "name": self.function_name or "",
                "arguments": self.arguments,
                "call_id": self.call_id or self.item_id,
            }
        return {
            "id": self.item_id,
            "type": "message",
            "role": Role.ASSISTANT.value,
            "status": self.status,
            "content": [
                {
                    "type": "output_text",
                    "text": self.text,
                    "annotations": [],
                }
            ],
        }

    def build_output_item_added_event(self, sequence: int) -> ResponseStreamEvent:
        payload: Dict[str, Any] = {
            "id": self.item_id,
            "type": "function_call" if self.kind == "function_call" else "message",
            "status": "in_progress",
        }
        if self.kind == "function_call":
            payload.update({
                "name": self.function_name or "",
                "arguments": "",
                "call_id": self.call_id or self.item_id,
            })
        else:
            payload.update({"role": Role.ASSISTANT.value, "content": []})

        return ResponseStreamEvent(
            event="response.output_item.added",
            data={
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": self.index,
                "item": payload,
            },
        )

    def build_content_part_added_event(self, sequence: int) -> ResponseStreamEvent:
        return ResponseStreamEvent(
            event="response.content_part.added",
            data={
                "type": "response.content_part.added",
                "sequence_number": sequence,
                "output_index": self.index,
                "content_index": 0,
                "item_id": self.item_id,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                },
            },
        )


def _ensure_multimodal_items(
    content: Iterable[MultimodalContentItem | dict[str, Any]]
) -> list[MultimodalContentItem]:
    items: list[MultimodalContentItem] = []
    for item in content:
        if isinstance(item, MultimodalContentItem):
            items.append(item)
        else:
            items.append(MultimodalContentItem.model_validate(item))
    return items


def _coerce_tool_call(tool_call: dict[str, Any], fallback_index: int) -> ToolCall:
    function_payload = tool_call.get("function") or {}
    name = function_payload.get("name") or tool_call.get("name") or "tool"
    arguments = function_payload.get("arguments") or tool_call.get("arguments") or ""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, separators=(",", ":"))
    elif isinstance(arguments, str):
        arguments = arguments
    else:
        arguments = json.dumps(arguments)

    call_id = tool_call.get("id") or tool_call.get("call_id") or str(uuid4())

    return ToolCall(
        id=call_id,
        type=ToolType.FUNCTION,
        function=FunctionCall(name=name, arguments=arguments),
    )


def _convert_input_item_to_chat_messages(
    item: Any, choice_index: int
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []

    if isinstance(item, ChatMessage):
        messages.append(item)
        return messages

    if isinstance(item, str):
        messages.append(ChatMessage(role=Role.USER, content=item))
        return messages

    if hasattr(item, "model_dump"):
        try:
            return _convert_input_item_to_chat_messages(item.model_dump(), choice_index)
        except Exception:  # pragma: no cover - fallback if model_dump not available
            pass

    if isinstance(item, dict):
        item_type = item.get("type")

        if item_type == "function_call":
            tool_call = _coerce_tool_call(item, choice_index)
            messages.append(
                ChatMessage(role=Role.ASSISTANT, tool_calls=[tool_call])
            )
            return messages

        if item_type == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output)
            messages.append(
                ChatMessage(
                    role=Role.TOOL,
                    content=output,
                    tool_call_id=item.get("call_id"),
                )
            )
            return messages

        if item_type == "message" or ("role" in item and "content" in item):
            role_value = item.get("role", "user")
            try:
                role = Role(role_value)
            except ValueError:
                role = Role.USER

            content = item.get("content")
            chat_content = _convert_response_content_to_chat_content(content)

            tool_calls_data = item.get("tool_calls") or []
            tool_calls: list[ToolCall] = []
            for idx, tc in enumerate(tool_calls_data):
                if isinstance(tc, dict):
                    tool_calls.append(_coerce_tool_call(tc, idx))

            messages.append(
                ChatMessage(
                    role=role,
                    content=chat_content,
                    tool_calls=tool_calls or None,
                )
            )
            return messages

    return messages


def _convert_input_to_chat_messages(input_value: Any) -> list[ChatMessage]:
    messages: list[ChatMessage] = []

    if input_value is None:
        return messages

    if isinstance(input_value, (list, tuple)):
        for idx, item in enumerate(input_value):
            messages.extend(_convert_input_item_to_chat_messages(item, idx))
        return messages

    if isinstance(input_value, (ChatMessage, str, dict)):
        messages.extend(_convert_input_item_to_chat_messages(input_value, 0))
        return messages

    return messages


def _convert_response_content_to_chat_content(
    content: str | list[MultimodalContentItem] | None,
) -> str | list[MultimodalContentItem] | None:
    if content is None or isinstance(content, str):
        return content

    text_segments: list[str] = []
    multimodal: list[MultimodalContentItem] = []

    for item in content:
        mm_item = (
            item
            if isinstance(item, MultimodalContentItem)
            else MultimodalContentItem.model_validate(item)
        )

        item_type = (mm_item.type or "").lower()

        if item_type in {"input_text", "text", "output_text"}:
            if mm_item.text:
                text_segments.append(mm_item.text)
        elif item_type in {"input_image", "image_url"} and mm_item.image_url is not None:
            multimodal.append(
                MultimodalContentItem(type="image_url", image_url=mm_item.image_url)
            )
        elif item_type == "input_audio" and mm_item.input_audio is not None:
            multimodal.append(
                MultimodalContentItem(type="input_audio", input_audio=mm_item.input_audio)
            )
        else:
            multimodal.append(mm_item)

    if not multimodal and text_segments:
        return "\n\n".join(text_segments)

    if text_segments:
        multimodal.extend(
            MultimodalContentItem(type="text", text=text) for text in text_segments
        )

    return multimodal or None


def response_request_to_chat_request(response_request: ResponseRequest) -> ChatCompletionRequest:
    payload = response_request.model_dump(exclude_none=True)

    instructions = payload.pop("instructions", None)
    max_output_tokens = payload.pop("max_output_tokens", None)
    text_config = payload.pop("text", None)
    payload.pop("input", None)

    chat_messages: list[ChatMessage] = []

    if instructions:
        chat_messages.append(ChatMessage(role=Role.SYSTEM, content=instructions))

    chat_messages.extend(_convert_input_to_chat_messages(response_request.input))

    payload["messages"] = chat_messages

    if isinstance(text_config, dict):
        text_format = text_config.get("format")
        response_format = _convert_text_format_to_chat_response_format(text_format)
        if response_format is not None:
            payload["response_format"] = response_format

    if "tools" in payload:
        payload["tools"] = _normalize_tools(payload["tools"])

    payload.setdefault("stream", response_request.stream)

    if max_output_tokens is not None:
        payload["max_completion_tokens"] = max_output_tokens

    return ChatCompletionRequest.model_validate(payload)


def _convert_text_format_to_chat_response_format(text_format: Any) -> dict[str, Any] | None:
    if not isinstance(text_format, dict):
        return None

    format_type = text_format.get("type")
    if format_type == "json_schema":
        # Responses API: {"type":"json_schema","name":...,"schema":{...},"strict":...}
        # Chat Completions API: {"type":"json_schema","json_schema":{...}}
        if "json_schema" in text_format and isinstance(text_format["json_schema"], dict):
            return {"type": "json_schema", "json_schema": text_format["json_schema"]}

        json_schema: dict[str, Any] = {}
        for key in ("name", "description", "schema", "strict"):
            if key in text_format:
                json_schema[key] = text_format[key]
        return {"type": "json_schema", "json_schema": json_schema}

    if format_type in {"text", "json_object"}:
        return {"type": format_type}

    return None


def _normalize_tools(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools

    normalized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized.append(tool)
            continue

        if "function" in tool or tool.get("type") != "function":
            normalized.append(tool)
            continue

        function_payload: dict[str, Any] = {}
        if "name" in tool:
            function_payload["name"] = tool["name"]
        if "description" in tool:
            function_payload["description"] = tool["description"]
        if "parameters" in tool:
            function_payload["parameters"] = tool["parameters"]

        normalized.append({"type": "function", "function": function_payload})

    return normalized


def _message_content_to_output_items(message: ChatMessage | None) -> list[dict[str, Any]]:
    if message is None or message.content is None:
        return [{"type": "output_text", "text": "", "annotations": []}]

    if isinstance(message.content, str):
        return [{"type": "output_text", "text": message.content, "annotations": []}]

    output_items: list[dict[str, Any]] = []
    if isinstance(message.content, list):
        for item in _ensure_multimodal_items(message.content):
            if item.type == "text" and item.text is not None:
                output_items.append(
                    {"type": "output_text", "text": item.text, "annotations": []}
                )

    if not output_items:
        output_items.append({"type": "output_text", "text": "", "annotations": []})

    return output_items


def _build_usage_dict(usage: ChatCompletionUsage | None) -> dict[str, Any]:
    if usage is None:
        return {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        }

    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0

    return {
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens,
    }


def chat_response_to_response(chat_response: ChatCompletionResponse) -> dict[str, Any]:
    response_id = chat_response.id
    created_at = float(chat_response.created)

    output_items: list[dict[str, Any]] = []
    for choice in chat_response.choices:
        message = choice.message
        role = message.role.value if message else Role.ASSISTANT.value

        if message and message.tool_calls:
            for idx, tool_call in enumerate(message.tool_calls):
                output_items.append(
                    {
                        "id": f"{response_id}-tool-{choice.index}-{idx}",
                        "type": "function_call",
                        "status": "completed",
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "call_id": tool_call.id,
                    }
                )

        content_items = _message_content_to_output_items(message)
        has_non_empty_text = any(
            isinstance(item, dict) and item.get("text") for item in content_items
        )
        if content_items and (has_non_empty_text or not (message and message.tool_calls)):
            output_items.append(
                {
                    "id": f"{response_id}-output-{choice.index}",
                    "type": "message",
                    "role": role,
                    "status": "completed",
                    "content": content_items,
                }
            )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": chat_response.model,
        "status": "completed",
        "output": output_items,
        "usage": _build_usage_dict(chat_response.usage),
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "metadata": None,
        "instructions": None,
        "reasoning": None,
        "service_tier": None,
        "previous_response_id": None,
        "prompt": None,
        "text": None,
        "background": None,
        "max_tool_calls": None,
        "prompt_cache_key": None,
        "safety_identifier": None,
    }


def _extract_text_from_delta(delta: ChatMessage | None) -> str:
    if not delta or not delta.content:
        return ""
    if isinstance(delta.content, str):
        return delta.content
    if isinstance(delta.content, list):
        text_parts: list[str] = []
        for item in delta.content:
            mm_item = item if isinstance(item, MultimodalContentItem) else MultimodalContentItem.model_validate(item)
            if mm_item.type == "text" and mm_item.text:
                text_parts.append(mm_item.text)
        return "".join(text_parts)
    return ""


class ResponseStreamAdapter:
    """Builds Responses API compliant streaming events from chat completion chunks."""

    def __init__(self, response_id: str, model: str) -> None:
        self.response_id = response_id
        self.model = model
        self._created_event_emitted = False
        self._created_at: float | None = None
        self._sequence = 0
        self._usage: ChatCompletionUsage | None = None
        self._items: dict[str, OutputItemState] = {}
        self._key_to_index: dict[str, int] = {}
        self._index_to_key: dict[int, str] = {}
        self._message_context: dict[int, str] = {}
        self._next_index = 0

    def _allocate_index(self, key: str) -> int:
        if key in self._key_to_index:
            return self._key_to_index[key]
        index = self._next_index
        self._next_index += 1
        self._key_to_index[key] = index
        self._index_to_key[index] = key
        return index

    def set_response_id(self, response_id: str) -> None:
        if self._created_event_emitted:
            return
        self.response_id = response_id

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _ensure_created_event(self) -> list[ResponseStreamEvent]:
        if self._created_event_emitted:
            return []

        created = self._created_at or time.time()
        self._created_at = created

        event = ResponseStreamEvent(
            event="response.created",
            data={
                "type": "response.created",
                "sequence_number": self._next_sequence(),
                "response": self._build_response_dict(status="in_progress", include_usage=False),
            },
        )
        self._created_event_emitted = True
        return [event]

    def _message_key(self, choice_index: int) -> str:
        context_key = self._message_context.get(choice_index)
        if context_key:
            return context_key
        key = f"choice-{choice_index}-message-{self._next_index}"
        self._message_context[choice_index] = key
        return key

    def _tool_call_key(self, choice_index: int, call_index: int, tool_call: Any) -> str:
        tool_id = getattr(tool_call, "id", None)
        if tool_id:
            return f"tool-{tool_id}"
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "unknown") if function else "unknown"
        return f"choice-{choice_index}-tool-{call_index}-{name}"

    def _ensure_message_item(
        self, choice_index: int
    ) -> tuple[list[ResponseStreamEvent], OutputItemState]:
        key = self._message_key(choice_index)
        state = self._items.get(key)
        if state:
            return [], state

        index = self._allocate_index(key)
        item_id = f"{self.response_id}-output-{index}"

        state = OutputItemState(
            key=key,
            choice_index=choice_index,
            index=index,
            item_id=item_id,
            kind="message",
        )
        self._items[key] = state
        self._message_context[choice_index] = key

        events = [
            state.build_output_item_added_event(self._next_sequence()),
            state.build_content_part_added_event(self._next_sequence()),
        ]

        return events, state

    def _append_text_delta(self, state: OutputItemState, text: str) -> str:
        existing_text = state.text
        if text.startswith(existing_text):
            delta_fragment = text[len(existing_text):]
            state.text = text
        else:
            delta_fragment = text
            state.text = existing_text + text
        return delta_fragment

    def _ensure_function_call_item(
        self, choice_index: int, call_index: int, tool_call: Any
    ) -> tuple[list[ResponseStreamEvent], OutputItemState]:
        key = self._tool_call_key(choice_index, call_index, tool_call)
        events: list[ResponseStreamEvent] = []
        state = self._items.get(key)
        function = getattr(tool_call, "function", None)
        call_id = getattr(tool_call, "id", None)
        name = getattr(function, "name", None) if function else None

        # Reset any pending assistant message for this choice index to ensure
        # subsequent text starts a fresh message item.
        existing_message_key = self._message_context.pop(choice_index, None)
        if existing_message_key and existing_message_key in self._items:
            existing_state = self._items[existing_message_key]
            if not existing_state.done_emitted:
                events.extend(self._emit_message_done(existing_state))

        if state:
            if name and not state.function_name:
                state.function_name = name
            if call_id and not state.call_id:
                state.call_id = call_id
            return events, state

        index = self._allocate_index(key)
        item_id = call_id or f"{self.response_id}-tool-{index}"

        state = OutputItemState(
            key=key,
            choice_index=choice_index,
            index=index,
            item_id=item_id,
            kind="function_call",
            function_name=name,
            call_id=call_id or item_id,
        )
        self._items[key] = state

        events.append(state.build_output_item_added_event(self._next_sequence()))

        return events, state

    def _build_text_delta_event(self, state: OutputItemState, delta_fragment: str) -> ResponseStreamEvent:
        return ResponseStreamEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "sequence_number": self._next_sequence(),
                "output_index": state.index,
                "content_index": 0,
                "item_id": state.item_id,
                "delta": delta_fragment,
                "logprobs": [],
            },
        )

    def on_chunk(self, chunk: ChatCompletionChunk) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        if chunk.created:
            self._created_at = float(chunk.created)

        events.extend(self._ensure_created_event())

        if chunk.usage:
            self._usage = chunk.usage

        for choice in chunk.choices:
            delta = choice.delta
            tool_calls = delta.tool_calls if delta and getattr(delta, "tool_calls", None) else []

            if tool_calls:
                text_before = _extract_text_from_delta(delta)
                if text_before:
                    new_events, message_state = self._ensure_message_item(choice.index)
                    events.extend(new_events)
                    delta_fragment = self._append_text_delta(message_state, text_before)
                    if delta_fragment:
                        events.append(self._build_text_delta_event(message_state, delta_fragment))
                    events.extend(self._emit_message_done(message_state))

                tool_states: list[OutputItemState] = []
                for call_idx, tool_call in enumerate(tool_calls):
                    new_events, state = self._ensure_function_call_item(choice.index, call_idx, tool_call)
                    events.extend(new_events)
                    tool_states.append(state)

                    function = getattr(tool_call, "function", None)
                    if function and getattr(function, "name", None) and not state.function_name:
                        state.function_name = function.name

                    arguments_delta = (
                        function.arguments if function and getattr(function, "arguments", None) else ""
                    )
                    if arguments_delta:
                        existing = state.arguments
                        if arguments_delta.startswith(existing):
                            delta_fragment = arguments_delta[len(existing):]
                            state.arguments = arguments_delta
                        else:
                            delta_fragment = arguments_delta
                            state.arguments = existing + arguments_delta

                        if not delta_fragment:
                            continue
                        events.append(
                            ResponseStreamEvent(
                                event="response.function_call_arguments.delta",
                                data={
                                    "type": "response.function_call_arguments.delta",
                                    "sequence_number": self._next_sequence(),
                                    "output_index": state.index,
                                    "item_id": state.item_id,
                                    "delta": delta_fragment,
                                },
                            )
                        )

                if choice.finish_reason:
                    for state in tool_states:
                        events.extend(self._emit_function_call_done(state))
                continue

            new_events, state = self._ensure_message_item(choice.index)
            events.extend(new_events)

            delta_text = _extract_text_from_delta(delta)
            if delta_text:
                delta_fragment = self._append_text_delta(state, delta_text)
                if delta_fragment:
                    events.append(self._build_text_delta_event(state, delta_fragment))

            if choice.finish_reason:
                events.extend(self._emit_message_done(state))

        return events

    def _emit_message_done(self, state: OutputItemState) -> list[ResponseStreamEvent]:
        if state.done_emitted:
            return []

        state.status = "completed"
        state.done_emitted = True

        current_key = self._message_context.get(state.choice_index)
        if current_key == state.key:
            self._message_context.pop(state.choice_index, None)

        events = [
            ResponseStreamEvent(
                event="response.output_text.done",
                data={
                    "type": "response.output_text.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "content_index": 0,
                    "item_id": state.item_id,
                    "text": state.text,
                    "logprobs": [],
                },
            )
        ]

        events.append(
            ResponseStreamEvent(
                event="response.content_part.done",
                data={
                    "type": "response.content_part.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "content_index": 0,
                    "item_id": state.item_id,
                    "part": {
                        "type": "output_text",
                        "text": state.text,
                        "annotations": [],
                    },
                },
            )
        )

        events.append(
            ResponseStreamEvent(
                event="response.output_item.done",
                data={
                    "type": "response.output_item.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "item": state.to_output_dict(),
                },
            )
        )

        return events

    def _emit_function_call_done(self, state: OutputItemState) -> list[ResponseStreamEvent]:
        if state.done_emitted:
            return []

        state.status = "completed"
        state.done_emitted = True

        events = [
            ResponseStreamEvent(
                event="response.function_call_arguments.done",
                data={
                    "type": "response.function_call_arguments.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "item_id": state.item_id,
                    "arguments": state.arguments,
                },
            )
        ]

        events.append(
            ResponseStreamEvent(
                event="response.output_item.done",
                data={
                    "type": "response.output_item.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "item": state.to_output_dict(),
                },
            )
        )

        return events

    def on_error(self, payload: dict[str, Any]) -> ResponseStreamEvent:
        message = payload.get("message", "Generation failed")
        code = payload.get("error")
        return ResponseStreamEvent(
            event="error",
            data={
                "type": "error",
                "sequence_number": self._next_sequence(),
                "message": message,
                "code": code,
                "param": None,
            },
        )

    def on_done(self) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        events.extend(self._ensure_created_event())

        if not self._items:
            new_events, state = self._ensure_message_item(0)
            events.extend(new_events)
            events.extend(self._emit_message_done(state))
        else:
            for index in sorted(self._index_to_key.keys()):
                key = self._index_to_key[index]
                state = self._items[key]
                if state.kind == "function_call":
                    events.extend(self._emit_function_call_done(state))
                else:
                    events.extend(self._emit_message_done(state))

        response = self._build_response_dict(status="completed", include_usage=True)
        events.append(
            ResponseStreamEvent(
                event="response.completed",
                data={
                    "type": "response.completed",
                    "sequence_number": self._next_sequence(),
                    "response": response,
                },
            )
        )
        return events

    def _build_response_dict(self, status: str, include_usage: bool) -> dict[str, Any]:
        created = self._created_at or time.time()
        output_items: list[dict[str, Any]] = []
        for index in sorted(self._index_to_key.keys()):
            key = self._index_to_key[index]
            state = self._items[key]
            output_items.append(state.to_output_dict())

        return {
            "id": self.response_id,
            "object": "response",
            "created_at": created,
            "model": self.model,
            "status": status,
            "output": output_items,
            "usage": _build_usage_dict(self._usage) if include_usage else None,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "temperature": None,
            "top_p": None,
            "max_output_tokens": None,
            "metadata": None,
            "instructions": None,
            "reasoning": None,
            "service_tier": None,
            "previous_response_id": None,
            "prompt": None,
            "text": None,
            "background": None,
            "max_tool_calls": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
