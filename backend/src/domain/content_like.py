from __future__ import annotations
from pydantic.dataclasses import dataclass


@dataclass
class ContentLike:
    content: str
