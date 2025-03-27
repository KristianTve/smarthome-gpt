from pydantic import dataclasses
from domain.message import Message


@dataclasses.dataclass
class ChatPayload:
    messages: list[Message]


@dataclasses.dataclass
class ChatResponse:
    answer: Message

