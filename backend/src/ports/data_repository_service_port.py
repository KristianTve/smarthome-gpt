from abc import ABC, abstractmethod

from domain.file_data import FileData
from domain.knowledge import Knowledge


class DataRepositoryServicePort(ABC):

    @abstractmethod
    def upload_knowledge(self, knowledge: list[Knowledge]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_file_data(self) -> list[FileData]:
        raise NotImplementedError

    @abstractmethod
    def initialize_data_structure(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_single_file_data(self, file_path: str) -> FileData:
        raise NotImplementedError
    