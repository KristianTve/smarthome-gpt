from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass
class FileData():
    file_bytes: bytes | None
    file_text: str | None
    source: str

    @staticmethod
    def create(
        file_bytes: bytes | None,
        file_text: str | None,
        source: str,
    ) -> FileData:
        return FileData(file_bytes=file_bytes, file_text=file_text, source=source)

