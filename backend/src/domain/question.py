from __future__ import annotations
from pydantic.dataclasses import dataclass
from domain.content_like import ContentLike


@dataclass
class Question(ContentLike):

    @staticmethod
    def create(content: str) -> Question:
        return Question(content=content)
