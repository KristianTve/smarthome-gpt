from fastapi import APIRouter
from api.inference import inference_router
from api.repository import repository_router

router = APIRouter(prefix="/api")
router.include_router(inference_router)
router.include_router(repository_router)
