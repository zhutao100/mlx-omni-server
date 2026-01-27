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
from ..chat.tool_loop_reasoning_cache import tool_loop_reasoning_cache
from ..utils.logger import logger
from .reasoning_envelope import ReasoningEnvelope, seal, unseal
from .schema import ResponseRequest, ResponseStreamEvent


@dataclass
class OutputItemState:
    key: str
    choice_index: int
    index: int
    item_id: str
    kind: Literal["message", "function_call", "reasoning"]
    status: str = "in_progress"
    text: str = ""
    arguments: str = ""
    function_name: Optional[str] = None
    call_id: Optional[str] = None
    encrypted_content: Optional[str] = None
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
        if self.kind == "reasoning":
            payload: Dict[str, Any] = {
                "id": self.item_id,
                "type": "reasoning",
                "status": self.status,
            }
            if self.encrypted_content is not None:
                payload["encrypted_content"] = self.encrypted_content
            return payload
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
        item_type = (
            "function_call"
            if self.kind == "function_call"
            else ("reasoning" if self.kind == "reasoning" else "message")
        )
        payload: Dict[str, Any] = {
            "id": self.item_id,
            "type": item_type,
            "status": "in_progress",
        }
        if self.kind == "function_call":
            payload.update({
                "name": self.function_name or "",
                "arguments": "",
                "call_id": self.call_id or self.item_id,
            })
        elif self.kind == "message":
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


def _parse_reasoning_envelope(item: Any) -> ReasoningEnvelope | None:
    candidate = item
    if hasattr(candidate, "model_dump"):
        try:
            candidate = candidate.model_dump()
        except Exception:  # pragma: no cover
            candidate = item

    if not isinstance(candidate, dict):
        return None

    if candidate.get("type") != "reasoning":
        return None

    encrypted_content = candidate.get("encrypted_content")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        return None

    try:
        return unseal(encrypted_content)
    except ValueError:
        return None


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

            if role == Role.ASSISTANT and not tool_calls and chat_content in (None, ""):
                # Some Responses clients (e.g. Codex CLI) can include empty assistant
                # message items between tool calls. Keeping them inserts extra template
                # tokens (e.g. `<|assistant|></think>`) and can cause minor prompt-cache
                # trims between rounds. Drop them as they carry no content.
                return []

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
        tool_call_id_to_message: dict[str, ChatMessage] = {}
        pending_reasoning: list[ReasoningEnvelope] = []

        def _has_non_empty_content(
            content: str | list[MultimodalContentItem] | None,
        ) -> bool:
            if content is None:
                return False
            if isinstance(content, str):
                return bool(content)
            return bool(content)

        def _merge_contents(
            existing: str | list[MultimodalContentItem] | None,
            incoming: str | list[MultimodalContentItem] | None,
        ) -> str | list[MultimodalContentItem] | None:
            if incoming is None:
                return existing
            if isinstance(incoming, str) and not incoming:
                return existing

            if existing is None:
                return incoming
            if isinstance(existing, str) and not existing:
                return incoming

            if isinstance(existing, str) and isinstance(incoming, str):
                return existing + incoming

            existing_items: list[MultimodalContentItem] = []
            if isinstance(existing, str):
                existing_items.append(MultimodalContentItem(type="text", text=existing))
            else:
                existing_items.extend(existing)

            if isinstance(incoming, str):
                existing_items.append(MultimodalContentItem(type="text", text=incoming))
            else:
                existing_items.extend(incoming)

            return existing_items

        for idx, item in enumerate(input_value):
            envelope = _parse_reasoning_envelope(item)
            if envelope is not None:
                pending_reasoning.append(envelope)
                for tool_call_id in envelope.tool_call_ids:
                    tool_loop_reasoning_cache.set(tool_call_id, envelope.reasoning)
                    seen_message = tool_call_id_to_message.get(tool_call_id)
                    if seen_message is not None and seen_message.reasoning is None:
                        seen_message.reasoning = envelope.reasoning
                continue

            new_messages = _convert_input_item_to_chat_messages(item, idx)
            for message in new_messages:
                appended = True
                target_message = message
                new_tool_calls = message.tool_calls or []

                if (
                    message.role == Role.ASSISTANT
                    and messages
                    and messages[-1].role == Role.ASSISTANT
                ):
                    prev = messages[-1]

                    should_merge = bool(message.tool_calls) or (
                        prev.tool_calls and _has_non_empty_content(message.content)
                    )
                    if should_merge:
                        appended = False
                        target_message = prev

                        if new_tool_calls:
                            prev.tool_calls = (prev.tool_calls or []) + list(new_tool_calls)

                        if _has_non_empty_content(message.content):
                            prev.content = _merge_contents(prev.content, message.content)

                        if message.reasoning is not None and prev.reasoning is None:
                            prev.reasoning = message.reasoning

                if appended:
                    messages.append(message)

                if target_message.role == Role.ASSISTANT and new_tool_calls:
                    for tool_call in new_tool_calls:
                        tool_call_id_to_message[tool_call.id] = target_message

                    if target_message.reasoning is None:
                        for pending in pending_reasoning:
                            if any(
                                tool_call.id in pending.tool_call_ids
                                for tool_call in new_tool_calls
                            ):
                                target_message.reasoning = pending.reasoning
                                break
        return messages

    if isinstance(input_value, (ChatMessage, str, dict)):
        envelope = _parse_reasoning_envelope(input_value)
        if envelope is not None:
            for tool_call_id in envelope.tool_call_ids:
                tool_loop_reasoning_cache.set(tool_call_id, envelope.reasoning)
            return messages

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

    payload.pop("include", None)
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
    if payload.get("stream"):
        stream_options = payload.get("stream_options")
        if isinstance(stream_options, dict):
            stream_options = dict(stream_options)
        else:
            stream_options = {}
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options

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

        tool_type = tool.get("type") or "function"
        if tool_type == "web_search":
            logger.warning("Dropping unsupported tool type 'web_search' from Responses request.")
            continue

        if "function" in tool:
            normalized.append(tool)
            continue

        if tool_type not in {"function", "custom"}:
            normalized.append(tool)
            continue

        function_payload: dict[str, Any] = {}
        if "name" in tool:
            function_payload["name"] = tool["name"]
        if "description" in tool:
            function_payload["description"] = tool["description"]
        if "parameters" in tool:
            function_payload["parameters"] = tool["parameters"]

        wrapped = {k: v for k, v in tool.items() if k not in {"name", "description", "parameters"}}
        wrapped["type"] = tool_type
        wrapped["function"] = function_payload
        normalized.append(wrapped)

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


def chat_messages_to_response_items(
    messages: list[ChatMessage],
    *,
    response_id: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    item_index = 0

    for message in messages:
        if message.role == Role.TOOL:
            output = message.content
            if not isinstance(output, str):
                output = json.dumps(output, separators=(",", ":"))
            items.append(
                {
                    "id": f"{response_id}-input-{item_index}",
                    "type": "function_call_output",
                    "status": "completed",
                    "call_id": message.tool_call_id,
                    "output": output,
                }
            )
            item_index += 1
            continue

        if message.role == Role.ASSISTANT and message.tool_calls:
            for tool_call in message.tool_calls:
                items.append(
                    {
                        "id": f"{response_id}-input-{item_index}",
                        "type": "function_call",
                        "status": "completed",
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "call_id": tool_call.id,
                    }
                )
                item_index += 1

        if message.role in {Role.USER, Role.SYSTEM}:
            content_list: list[dict[str, Any]] = []
            if isinstance(message.content, str):
                content_list.append({"type": "input_text", "text": message.content})
            elif isinstance(message.content, list):
                for content_item in _ensure_multimodal_items(message.content):
                    if content_item.type == "text" and content_item.text is not None:
                        content_list.append({"type": "input_text", "text": content_item.text})
                    elif content_item.type == "image_url" and content_item.image_url is not None:
                        content_list.append(
                            {
                                "type": "input_image",
                                "image_url": content_item.image_url.url,
                                "detail": "auto",
                            }
                        )
                    else:
                        content_list.append(
                            {"type": "input_text", "text": f"[{content_item.type}]"}
                        )
            else:
                content_list.append({"type": "input_text", "text": ""})

            items.append(
                {
                    "id": f"{response_id}-input-{item_index}",
                    "type": "message",
                    "status": "completed",
                    "role": message.role.value,
                    "content": content_list,
                }
            )
            item_index += 1
            continue

        if message.role == Role.ASSISTANT:
            items.append(
                {
                    "id": f"{response_id}-input-{item_index}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": _message_content_to_output_items(message),
                }
            )
            item_index += 1

    return items


def response_output_items_to_chat_messages(
    output_items: Iterable[dict[str, Any]],
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    tool_call_id_to_message: dict[str, ChatMessage] = {}
    pending_reasoning: list[ReasoningEnvelope] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "reasoning":
            envelope = _parse_reasoning_envelope(item)
            if envelope is not None:
                pending_reasoning.append(envelope)
                for tool_call_id in envelope.tool_call_ids:
                    tool_loop_reasoning_cache.set(tool_call_id, envelope.reasoning)
                    seen_message = tool_call_id_to_message.get(tool_call_id)
                    if seen_message is not None and seen_message.reasoning is None:
                        seen_message.reasoning = envelope.reasoning
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or str(uuid4())
            name = item.get("name") or "tool"
            arguments = item.get("arguments") or ""
            tool_call = ToolCall(
                id=call_id,
                type=ToolType.FUNCTION,
                function=FunctionCall(name=name, arguments=arguments),
            )

            if messages and messages[-1].role == Role.ASSISTANT:
                message = messages[-1]
                message.tool_calls = (message.tool_calls or []) + [tool_call]
            else:
                message = ChatMessage(role=Role.ASSISTANT, tool_calls=[tool_call])
                messages.append(message)

            tool_call_id_to_message[call_id] = message
            if message.reasoning is None:
                for pending in pending_reasoning:
                    if call_id in pending.tool_call_ids:
                        message.reasoning = pending.reasoning
                        break
            continue

        if item_type == "message":
            content = item.get("content") or []
            text_segments: list[str] = []
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "output_text"
                        and part.get("text")
                    ):
                        text_segments.append(str(part["text"]))
            if not text_segments:
                continue
            content_text = "".join(text_segments)

            if messages and messages[-1].role == Role.ASSISTANT and messages[-1].tool_calls:
                message = messages[-1]
                if message.content is None:
                    message.content = content_text
                elif isinstance(message.content, str):
                    message.content += content_text
                elif isinstance(message.content, list):
                    message.content = list(message.content) + [
                        MultimodalContentItem(type="text", text=content_text)
                    ]
                else:
                    message.content = content_text
            else:
                messages.append(ChatMessage(role=Role.ASSISTANT, content=content_text))

    return messages


def build_history_messages_for_next_request(
    *,
    input_messages: list[ChatMessage],
    instructions: str | None,
    output_items: list[dict[str, Any]],
) -> list[ChatMessage]:
    messages = list(input_messages)
    if (
        instructions
        and messages
        and messages[0].role == Role.SYSTEM
        and messages[0].content == instructions
    ):
        messages = messages[1:]

    messages.extend(response_output_items_to_chat_messages(output_items))
    return messages


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
    reasoning_tokens = (
        usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0
    )
    reasoning_tokens = max(0, min(int(reasoning_tokens), int(usage.completion_tokens)))

    return {
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": usage.total_tokens,
    }


def build_response_dict(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    request_echo: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_echo = request_echo or {}

    tools = request_echo.get("tools") or []
    if not isinstance(tools, list):
        tools = [tools]

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "error": error,
        "incomplete_details": incomplete_details,
        "model": model,
        "status": status,
        "output": output,
        "usage": usage,
        "parallel_tool_calls": bool(request_echo.get("parallel_tool_calls", True)),
        "tool_choice": request_echo.get("tool_choice", "auto"),
        "tools": tools,
        "temperature": request_echo.get("temperature"),
        "top_p": request_echo.get("top_p"),
        "truncation": request_echo.get("truncation"),
        "store": request_echo.get("store"),
        "max_output_tokens": request_echo.get("max_output_tokens"),
        "metadata": request_echo.get("metadata"),
        "instructions": request_echo.get("instructions"),
        "reasoning": request_echo.get("reasoning"),
        "service_tier": request_echo.get("service_tier"),
        "previous_response_id": request_echo.get("previous_response_id"),
        "prompt": request_echo.get("prompt"),
        "text": request_echo.get("text"),
        "background": request_echo.get("background"),
        "max_tool_calls": request_echo.get("max_tool_calls"),
        "prompt_cache_key": request_echo.get("prompt_cache_key"),
        "safety_identifier": request_echo.get("safety_identifier"),
        "top_logprobs": request_echo.get("top_logprobs"),
        "user": request_echo.get("user"),
    }


def chat_response_to_response(
    chat_response: ChatCompletionResponse,
    *,
    request_echo: dict[str, Any] | None = None,
    response_id_override: str | None = None,
) -> dict[str, Any]:
    response_id = response_id_override or chat_response.id
    created_at = int(chat_response.created)
    include = (request_echo or {}).get("include") or []
    include_reasoning_encrypted = (
        isinstance(include, list) and "reasoning.encrypted_content" in include
    )

    output_items: list[dict[str, Any]] = []
    for choice in chat_response.choices:
        message = choice.message
        role = message.role.value if message else Role.ASSISTANT.value

        if message and message.reasoning:
            item: dict[str, Any] = {
                "id": f"{response_id}-reasoning-{choice.index}",
                "type": "reasoning",
                "status": "completed",
            }
            if include_reasoning_encrypted:
                tool_call_ids = [
                    tool_call.id
                    for tool_call in (message.tool_calls or [])
                    if tool_call.id is not None
                ]
                item["encrypted_content"] = seal(
                    ReasoningEnvelope(
                        model=chat_response.model,
                        created_at=created_at,
                        tool_call_ids=tool_call_ids,
                        reasoning=message.reasoning,
                    )
                )
            output_items.append(item)

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

    return build_response_dict(
        response_id=response_id,
        created_at=created_at,
        model=chat_response.model,
        status="completed",
        output=output_items,
        usage=_build_usage_dict(chat_response.usage),
        request_echo=request_echo,
    )


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

    def __init__(
        self, response_id: str, model: str, *, request_echo: dict[str, Any] | None = None
    ) -> None:
        self.response_id = response_id
        self.model = model
        self._request_echo = request_echo or {}
        self._created_event_emitted = False
        self._in_progress_event_emitted = False
        self._created_at: int | None = None
        self._sequence = 0
        self._usage: ChatCompletionUsage | None = None
        self._error: dict[str, Any] | None = None
        self._items: dict[str, OutputItemState] = {}
        self._key_to_index: dict[str, int] = {}
        self._index_to_key: dict[int, str] = {}
        self._message_context: dict[int, str] = {}
        self._next_index = 0
        self._reasoning_by_choice: dict[int, str] = {}
        self._tool_call_ids_by_choice: dict[int, list[str]] = {}
        self._tool_call_ids_seen_by_choice: dict[int, set[str]] = {}
        include = self._request_echo.get("include") or []
        self._include_reasoning_encrypted = (
            isinstance(include, list) and "reasoning.encrypted_content" in include
        )

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

        created = self._created_at or int(time.time())
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

    def _ensure_in_progress_event(self) -> list[ResponseStreamEvent]:
        if self._in_progress_event_emitted:
            return []
        if not self._created_event_emitted:
            return []

        event = ResponseStreamEvent(
            event="response.in_progress",
            data={
                "type": "response.in_progress",
                "sequence_number": self._next_sequence(),
                "response": self._build_response_dict(status="in_progress", include_usage=False),
            },
        )
        self._in_progress_event_emitted = True
        return [event]

    def _ensure_lifecycle_started(self) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        events.extend(self._ensure_created_event())
        events.extend(self._ensure_in_progress_event())
        return events

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

    @staticmethod
    def _merge_accumulated_text(existing: str, incoming: str) -> str:
        if not existing:
            return incoming
        if not incoming:
            return existing
        if incoming.startswith(existing):
            return incoming
        if existing.startswith(incoming):
            return existing
        return existing + incoming

    def _track_tool_call_id(self, choice_index: int, call_id: str | None) -> None:
        if not call_id:
            return
        seen = self._tool_call_ids_seen_by_choice.setdefault(choice_index, set())
        if call_id in seen:
            return
        seen.add(call_id)
        self._tool_call_ids_by_choice.setdefault(choice_index, []).append(call_id)

    def _ensure_reasoning_item(
        self, choice_index: int
    ) -> tuple[list[ResponseStreamEvent], OutputItemState]:
        key = f"choice-{choice_index}-reasoning"
        state = self._items.get(key)
        if state:
            return [], state

        index = self._allocate_index(key)
        item_id = f"{self.response_id}-reasoning-{choice_index}"
        state = OutputItemState(
            key=key,
            choice_index=choice_index,
            index=index,
            item_id=item_id,
            kind="reasoning",
        )
        self._items[key] = state

        events = [state.build_output_item_added_event(self._next_sequence())]
        return events, state

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
            self._created_at = int(chunk.created)

        events.extend(self._ensure_lifecycle_started())

        if chunk.usage:
            self._usage = chunk.usage

        for choice in chunk.choices:
            delta = choice.delta
            delta_reasoning = getattr(delta, "reasoning", None) if delta else None
            if isinstance(delta_reasoning, str) and delta_reasoning:
                existing_reasoning = self._reasoning_by_choice.get(choice.index, "")
                self._reasoning_by_choice[choice.index] = self._merge_accumulated_text(
                    existing_reasoning,
                    delta_reasoning,
                )

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
                    self._track_tool_call_id(choice.index, state.call_id)

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

            delta_text = _extract_text_from_delta(delta)
            existing_key = self._message_context.get(choice.index)
            existing_state = self._items.get(existing_key) if existing_key else None

            # Avoid creating empty assistant message items (e.g. when the upstream chat
            # stream emits an empty delta such as the `</think>` boundary). Empty message
            # output items can get echoed back by clients and introduce prompt/template
            # drift between rounds.
            if not delta_text and existing_state is None:
                continue

            new_events, state = self._ensure_message_item(choice.index)
            events.extend(new_events)

            if delta_text:
                delta_fragment = self._append_text_delta(state, delta_text)
                if delta_fragment:
                    events.append(self._build_text_delta_event(state, delta_fragment))

            if choice.finish_reason:
                events.extend(self._emit_message_done(state))

        return events

    def _emit_reasoning_done(self, state: OutputItemState) -> list[ResponseStreamEvent]:
        if state.done_emitted:
            return []

        state.status = "completed"
        state.done_emitted = True

        return [
            ResponseStreamEvent(
                event="response.output_item.done",
                data={
                    "type": "response.output_item.done",
                    "sequence_number": self._next_sequence(),
                    "output_index": state.index,
                    "item": state.to_output_dict(),
                },
            )
        ]

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

    def on_error(self, payload: dict[str, Any]) -> list[ResponseStreamEvent]:
        message = payload.get("message", "Generation failed")
        raw_code = payload.get("error")
        if raw_code in {None, "Generation failed"}:
            code = "server_error"
            error_type = "server_error"
        elif raw_code == "Request cancelled":
            code = "cancelled"
            error_type = "cancelled"
        else:
            code = str(raw_code)
            error_type = "server_error"

        self._error = {
            "message": message,
            "type": error_type,
            "code": code,
            "param": None,
        }

        events: list[ResponseStreamEvent] = []
        events.extend(self._ensure_lifecycle_started())
        events.append(
            ResponseStreamEvent(
                event="error",
                data={
                    "type": "error",
                    "sequence_number": self._next_sequence(),
                    "message": message,
                    "code": code,
                    "param": None,
                },
            )
        )
        return events

    def on_done(self) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        events.extend(self._ensure_lifecycle_started())

        created = self._created_at or int(time.time())
        for choice_index, reasoning in sorted(self._reasoning_by_choice.items()):
            if not reasoning:
                continue
            new_events, state = self._ensure_reasoning_item(choice_index)
            events.extend(new_events)
            if self._include_reasoning_encrypted:
                tool_call_ids = self._tool_call_ids_by_choice.get(choice_index, [])
                state.encrypted_content = seal(
                    ReasoningEnvelope(
                        model=self.model,
                        created_at=created,
                        tool_call_ids=list(tool_call_ids),
                        reasoning=reasoning,
                    )
                )

        if self._items:
            for index in sorted(self._index_to_key.keys()):
                key = self._index_to_key[index]
                state = self._items[key]
                if state.kind == "function_call":
                    events.extend(self._emit_function_call_done(state))
                elif state.kind == "message":
                    events.extend(self._emit_message_done(state))
                else:
                    events.extend(self._emit_reasoning_done(state))
        elif self._error is None:
            new_events, state = self._ensure_message_item(0)
            events.extend(new_events)
            events.extend(self._emit_message_done(state))

        status = "failed" if self._error else "completed"
        response = self._build_response_dict(status=status, include_usage=True, error=self._error)
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

    def _build_response_dict(
        self,
        status: str,
        include_usage: bool,
        *,
        error: dict[str, Any] | None = None,
        incomplete_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created = self._created_at or int(time.time())
        output_items: list[dict[str, Any]] = []
        for index in sorted(self._index_to_key.keys()):
            key = self._index_to_key[index]
            state = self._items[key]
            output_items.append(state.to_output_dict())

        return build_response_dict(
            response_id=self.response_id,
            created_at=created,
            model=self.model,
            status=status,
            output=output_items,
            usage=_build_usage_dict(self._usage) if include_usage else None,
            request_echo=self._request_echo,
            error=error,
            incomplete_details=incomplete_details,
        )
