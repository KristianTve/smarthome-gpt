from typing import Any
from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from aiohttp import ClientSession
import os

API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("SELF_API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

class AuthError(Exception):
    def __init__(self, error: dict[str, str], status_code: int, *args: object):
        super().__init__(*args)
        self.error = error
        self.status_code = status_code

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=401, detail={"code": "invalid_api_key", "description": "Invalid API Key"}
    )

async def send_post_request(url: str, data: dict[str, str]):
    async with ClientSession() as session:
        async with session.post(url, json=data) as response:
            response_data: dict[str, Any] = await response.json()
            print(f"Response data: {response_data}")
            return response_data

auth_dependency: str = Depends(get_api_key)
