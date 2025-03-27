from fastapi import APIRouter, Response, status, BackgroundTasks
import traceback

from config.auth import auth_dependency
from domain.knowledge import Knowledge
from api.repository.dependencies import (
    data_repository_dependency,
    data_processing_dependency,
)
from ports.data_repository_service_port import DataRepositoryServicePort

repository_router = APIRouter(prefix="/repository")


@repository_router.get("/update_knowledge")
def ingest(
    response: Response,
    background_tasks: BackgroundTasks,
    data_repository_service=data_repository_dependency,
    data_processing_service=data_processing_dependency,
    # _: dict[str, str] = auth_dependency,
) -> dict[str, str]:
    try:
        # Define a background task function
        def run_ingestion():
            try:
                knowledge: list[Knowledge] = data_processing_service.generate_knowledge()
                data_repository_service.initialize_data_structure()
                data_repository_service.upload_knowledge(knowledge)
            except Exception:
                traceback.print_exc()

        background_tasks.add_task(run_ingestion)

        response.status_code = status.HTTP_200_OK
        return {"message": "Successfully commenced ingestion process"}

    except Exception:
        traceback.print_exc()
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Failed to initiate knowledge processing"}



@repository_router.get("/delete_caches")
def delete_caches(
    response: Response,
    data_repository_service: DataRepositoryServicePort = data_repository_dependency,
    # _: dict[str, str] = auth_dependency,
):
    try:
        data_repository_service.delete_caches()
        response.status_code = status.HTTP_200_OK
        return {"status": "deleted"}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": "Failed to flush caches"}
