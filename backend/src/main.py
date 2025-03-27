from fastapi import FastAPI, Request

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api import router
from config.auth import AuthError

app = FastAPI()
app.include_router(router)


# Fix CORS issue
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthError)
async def auth_exception_handler(_: Request, exc: AuthError):
    return JSONResponse(status_code=exc.status_code, content=exc.error)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5200)
