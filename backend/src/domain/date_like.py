from __future__ import annotations
from datetime import datetime

from pydantic import field_validator
from pydantic.dataclasses import dataclass


@dataclass
class DateLike:
    created_at: str

    @field_validator("created_at")
    def validate_created_at(cls, value: str | None):
        try:
            if value is None:
                return value
            datetime.strptime(value, "%d/%m/%YT%H:%M:%S.%f")
            return value
        except Exception as e:
            raise ValueError("Invalid date format in DateLike creation", e)
