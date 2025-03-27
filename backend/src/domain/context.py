from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass
class Context:
    content: str

    @staticmethod
    def create(content: str) -> Context:
        return Context(content=content)
