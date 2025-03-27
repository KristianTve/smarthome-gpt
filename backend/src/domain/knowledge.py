from __future__ import annotations
from typing_extensions import TypedDict
from pydantic.dataclasses import dataclass

from domain.content_like import ContentLike
from domain.context import Context

class KnowledgeDict(TypedDict):
    id: str
    content: str
    source: str
    content_vector: list[float] | None


@dataclass
class Knowledge(ContentLike):
    id: str
    embedding: list[float] | None
    chunk_size: int | None
    source: str

    @staticmethod
    def create(
        id: str,
        content: str,
        embedding: list[float] | None,
        chunk_size: int | None,
        source: str,
    ) -> Knowledge:
        return Knowledge(
            id=id,
            content=content,
            embedding=embedding,
            chunk_size=chunk_size,
            source=source,
        )

    def to_dict(self) -> KnowledgeDict:
        """Convert Knowledge instance to a JSON-serializable dictionary."""
        # Serialize image objects too
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "content_vector": self.embedding,
        }


@dataclass
class FormattedKnowledge:
    contexts: list[Context]
    sources: list[str]

    @staticmethod
    def create(
        contexts: list[Context], sources: list[str]
    ) -> FormattedKnowledge:
        return FormattedKnowledge(contexts, sources)
