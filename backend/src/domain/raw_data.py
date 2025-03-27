from __future__ import annotations
from pydantic.dataclasses import dataclass

from domain.content_like import ContentLike


@dataclass
class RawData(ContentLike):
    source: str

    @staticmethod
    def create(content: str, source: str) -> RawData:
        return RawData(content=content, source=source)
