from __future__ import annotations

from pathlib import Path

from jinja2 import Environment


def test_qwen3_5_template_renders_non_first_system_messages() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    template_path = repo_root / "src/mlx_omni_server/chat/templates/qwen3_5_chat_template.jinja"
    template_text = template_path.read_text(encoding="utf-8")

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    env = Environment()
    env.globals["raise_exception"] = raise_exception

    template = env.from_string(template_text)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {
            "role": "system",
            "content": "<AUTO_COMPACT_WORK_NOTES_REQUEST>\nRespond with a work note.",
        },
    ]
    tools = [{"type": "function", "function": {"name": "noop", "arguments": {}}}]

    rendered = template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        num_images=0,
        num_audios=0,
        add_vision_id=False,
    )

    assert "<AUTO_COMPACT_WORK_NOTES_REQUEST>" in rendered
