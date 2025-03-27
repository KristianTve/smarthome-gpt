from __future__ import annotations
from pydantic import dataclasses

@dataclasses.dataclass
class Message():
    role: str
    content: str

    @staticmethod
    def create(
        role: str,
        content: str,
    ) -> Message:
        return Message(
            role=role,
            content=content,
        )
