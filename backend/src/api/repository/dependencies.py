from fastapi import Depends

from config.dependencies import (
    get_data_processing_service,
    get_data_repository,
)
from ports.data_processing_service_port import DataProcessingServicePort
from ports.data_repository_service_port import DataRepositoryServicePort


data_repository_dependency: DataRepositoryServicePort = Depends(get_data_repository)
data_processing_dependency: DataProcessingServicePort = Depends(get_data_processing_service)