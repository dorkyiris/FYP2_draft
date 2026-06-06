"""Optional API key authentication via REHAB_API_KEY env var."""

import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(_api_key_header)) -> str | None:
    """Allow request if no key is configured, or if provided key matches."""
    configured = os.environ.get("REHAB_API_KEY")
    if not configured:
        return None  # auth disabled
    if api_key == configured:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key",
    )
