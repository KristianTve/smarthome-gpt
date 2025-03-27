from fastapi import Depends

from config.dependencies import (
    get_data_processing_service,
    get_inference_service,
    get_retrieval_service,
)
from ports.data_processing_service_port import DataProcessingServicePort
from ports.inference_service_port import InferenceServicePort
from ports.retrieval_service_port import RetrievalServicePort

inference_dependency: InferenceServicePort = Depends(get_inference_service)
retrieval_dependency: RetrievalServicePort = Depends(get_retrieval_service)
data_processing_dependency: DataProcessingServicePort = Depends(
    get_data_processing_service
)
