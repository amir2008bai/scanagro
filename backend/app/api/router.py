from fastapi import APIRouter

from app.api.routes import detections, exports, health, images, jobs

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(images.router)
api_router.include_router(jobs.router)
api_router.include_router(detections.router)
api_router.include_router(exports.router)
