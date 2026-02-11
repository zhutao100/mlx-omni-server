from __future__ import annotations

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel


def extract_extra_params(model: BaseModel, standard_fields: Collection[str]) -> dict[str, Any]:
    """Return any request fields that aren't part of the standard API surface."""
    dump = model.model_dump()
    return {k: v for k, v in dump.items() if k not in standard_fields}
