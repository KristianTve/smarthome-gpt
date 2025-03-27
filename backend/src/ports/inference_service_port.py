from abc import ABC, abstractmethod

from domain.answer import Answer
from domain.message import Message
from domain.query import Query
from domain.file_data import FileData


class InferenceServicePort(ABC):

    @abstractmethod
    def generate_answer(
        self,
        query: Query,
    ) -> Message:
        raise NotImplementedError
