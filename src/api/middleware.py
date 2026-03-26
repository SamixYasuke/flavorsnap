from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from .models import APISettings


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        requests_per_window: int,
        window_seconds: int,
        exempt_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.exempt_paths = set(exempt_paths or [])
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.exempt_paths:
            return await call_next(request)

        limited, remaining, retry_after = self._register(request)
        if limited:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    def _register(self, request: Request) -> tuple[bool, int, int]:
        key = self._client_key(request)
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.requests_per_window:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return True, 0, retry_after

            bucket.append(now)
            remaining = self.requests_per_window - len(bucket)
            return False, remaining, 0

    @staticmethod
    def _client_key(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
        return "anonymous"


def configure_middleware(app: FastAPI, settings: APISettings) -> None:
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        requests_per_window=settings.rate_limit.requests,
        window_seconds=settings.rate_limit.window_seconds,
        exempt_paths=settings.rate_limit.exempt_paths,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
