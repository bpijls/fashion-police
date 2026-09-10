from __future__ import annotations

import httpx


class ComputeError(RuntimeError):
    """Compute tier was unreachable or returned an error."""

    def __init__(self, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class ComputeClient:
    def __init__(self, base_url: str, secret: str, timeout_s: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-Compute-Secret": secret} if secret else {}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthz(self) -> dict:
        r = await self._client.get(f"{self._base_url}/healthz")
        r.raise_for_status()
        return r.json()

    async def labels(self) -> dict:
        try:
            r = await self._client.get(f"{self._base_url}/labels")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ComputeError(f"compute /labels failed: {exc}") from exc
        return r.json()

    async def infer(self, image_bytes: bytes, content_type: str) -> dict:
        files = {"frame": ("frame", image_bytes, content_type or "application/octet-stream")}
        try:
            r = await self._client.post(
                f"{self._base_url}/infer", files=files, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise ComputeError("compute timed out", status=504) from exc
        except httpx.HTTPError as exc:
            raise ComputeError(f"compute unreachable: {exc}", status=502) from exc

        if r.status_code == 401:
            raise ComputeError("compute rejected the shared secret", status=500)
        if r.status_code >= 500:
            raise ComputeError(f"compute error {r.status_code}", status=502)
        if r.status_code >= 400:
            raise ComputeError(f"compute rejected the image ({r.status_code})", status=400)
        return r.json()
