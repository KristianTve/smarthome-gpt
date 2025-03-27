from __future__ import annotations
from pydantic.dataclasses import dataclass

from domain.content_like import ContentLike


@dataclass
class Answer(ContentLike):

    @staticmethod
    def create(content: str) -> Answer:
        return Answer(content=content)
