from abc import ABC, abstractmethod

from domain.knowledge import Knowledge


class RetrievalServicePort(ABC):
    @abstractmethod
    def find_text_context(self, text: str) -> list[Knowledge]:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_sensor_data(self) -> object:
        raise NotImplementedError