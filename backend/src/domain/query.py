from __future__ import annotations
from pydantic.dataclasses import dataclass

from domain.context import Context
from domain.message import Message
from domain.question import Question
from domain.system_prompt import SystemPrompt


@dataclass
class Query:
    system_prompt: SystemPrompt | str
    contexts: list[Context]
    question: Question
    history: list[Message]

    @staticmethod
    def create(
        system_prompt: SystemPrompt | str,
        contexts: list[Context],
        question: Question,
        history: list[Message],
    ) -> Query:
        return Query(
            system_prompt=system_prompt,
            contexts=contexts,
            question=question,
            history=history,
        )
