from abc import ABC, abstractmethod

from domain.knowledge import Knowledge, FormattedKnowledge


class DataProcessingServicePort(ABC):

    @abstractmethod
    def generate_knowledge(self) -> list[Knowledge]:
        raise NotImplementedError

    @abstractmethod
    def format_knowledge(self, docs: list[Knowledge], sensor_data: str) -> FormattedKnowledge:
        raise NotImplementedError
