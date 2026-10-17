"""The one error shape every endpoint returns (CLAUDE.md 10):

    {"error": {"code": "...", "detail": "..."}}

FastAPI's defaults would wrap it as {"detail": {...}} for raised errors and
return {"detail": [...]} for validation failures; these handlers keep the
documented contract instead.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def api_error(status: int, code: str, detail: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "detail": detail})


def install(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            body = {"code": str(detail["code"]), "detail": str(detail.get("detail", ""))}
        else:
            body = {"code": _code_for(error.status_code), "detail": str(detail)}
        return JSONResponse({"error": body}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        problems = error.errors()
        first = problems[0] if problems else {}
        # The field that failed, so a client can point at it, without echoing
        # the submitted values -- one of which may be a password.
        field = str(first.get("loc", ("",))[-1]) if first else ""
        message = str(first.get("msg", "invalid request"))
        return JSONResponse(
            {"error": {"code": "INVALID_REQUEST", "detail": f"{field}: {message}", "field": field}},
            status_code=422,
        )


def _code_for(status: int) -> str:
    return {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(status, f"HTTP_{status}")
