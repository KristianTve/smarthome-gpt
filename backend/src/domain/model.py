from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass
class Model:
    name: str

    @staticmethod
    def create(name: str) -> Model:
        return Model(name=name)
