from __future__ import annotations
from typing import TypedDict
from pydantic.dataclasses import dataclass
from domain.context import Context


@dataclass
class Image:
    tag: str  # Tag for identifying the image, e.g., "[IMAGE:1]"
    img_bytes: bytes  # Raw image bytes (to be base64 encoded later if needed)
    img_format: str  # Image format, e.g., "png", "jpg"
    context: Context | None

    @staticmethod
    def create(
        tag: str,
        img_bytes: bytes,
        img_format: str,
        context: Context | None = None,
    ) -> Image:
        return Image(tag, img_bytes, img_format, context)
    
    def to_dict(self) -> ImageData:
        return {
            "tag": self.tag,
            "image_bytes": self.img_bytes,
            "image_format": self.img_format,
        }

class ImageData(TypedDict):
    tag: str
    image_bytes: bytes
    image_format: str


class ImageHash(TypedDict):
    hash: str
    img: Image
