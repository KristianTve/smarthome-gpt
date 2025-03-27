import os

from adapters.data_repository_service import DataRepositoryService
from adapters.inference_service import InferenceService
from adapters.retrieval_service import RetrievalService
from adapters.data_processing_service import DataProcessingService
from config import config
from domain.model import Model
from ports.data_processing_service_port import DataProcessingServicePort
from ports.data_repository_service_port import DataRepositoryServicePort
from ports.inference_service_port import InferenceServicePort
from ports.retrieval_service_port import RetrievalServicePort


def get_retrieval_service() -> RetrievalServicePort:
    db_connection_string = os.environ.get("DB_CONNECTION_STRING")
    return RetrievalService(
        db_connection_string=db_connection_string,
        collection_name=config.collection_name,
        n_chunks_retrieved=config.n_chunks_retrieved,
        embedding_api_version=config.embedding_api_version,
        embedding_model_name=Model.create(name=config.embedding_model_name),
    )
    

def get_inference_service() -> InferenceServicePort:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    return InferenceService(
        model_deployment=config.model_deployment,
        temperature=config.temperature,
        max_tokens=config.max_tokens_to_sample,
        openai_api_key=openai_api_key,
    )


def get_data_repository() -> DataRepositoryServicePort:
    db_connection_string = os.environ.get("DB_CONNECTION_STRING")
    if db_connection_string is None:
        raise KeyError("ERROR: DB connection string not found in environment variables")

    return DataRepositoryService(
        collection_name=config.collection_name,
        db_connection_string=db_connection_string,
        embedding_dimension_size=config.embedding_dimension_size,
        pdf_folder=config.pdf_folder
    )

def get_data_processing_service() -> DataProcessingServicePort:
    return DataProcessingService(
        retrieval_service=get_retrieval_service(),
        data_repository_service=get_data_repository(),
        inference_service=get_inference_service(),
        chunk_size=config.chunk_size,
        embedding_model_name=config.embedding_model_name,
        embedding_dimension_size=config.embedding_dimension_size,
        chunk_overlap=config.chunk_overlap,
        pdf_folder=config.pdf_folder,
    )
