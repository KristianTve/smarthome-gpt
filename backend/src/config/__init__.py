import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class Config(BaseModel):
    # Retrieval
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    n_chunks_retrieved: int
    embedding_model_name: str
    embedding_api_version: str
    embedding_dimension_size: int

    # Data Repository
    pdf_folder: str

    # Inference
    model_deployment: str
    temperature: float
    inference_api_version: str
    max_tokens_to_sample: int


def load_config() -> Config:
    # Get the directory of the current script
    current_script_directory = Path(__file__).parent
    file = os.path.join(current_script_directory, "config.yml")

    if not os.path.exists(file):
        raise RuntimeError(f"No config file exists at {file}.")

    with open(file, "r") as file:
        parsed_yaml: dict[str, Any] = yaml.safe_load(file)
        return Config(**parsed_yaml)  # pyright: ignore[reportAny]


config = load_config()
_ = load_dotenv()
