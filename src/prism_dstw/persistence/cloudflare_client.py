"""Async Cloudflare Manifold client with concurrent R2 multipart uploads."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Self, Sequence, TypedDict, cast

import httpx


JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

DEFAULT_CHUNK_SIZE = 50 * 1024 * 1024
DEFAULT_CONCURRENCY = 16


class CloudflareClientError(RuntimeError):
    """Raised when the PRISM Cloudflare API returns malformed data."""


class CompletedPart(TypedDict):
    part_number: int
    etag: str


class MultipartUploadResult(TypedDict):
    key: str
    upload_id: str
    part_count: int
    parts: list[CompletedPart]
    worker_response: JsonObject


@dataclass(frozen=True)
class UploadPart:
    part_number: int
    data: bytes


def _env_required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise CloudflareClientError(f"missing required environment variable: {name}")
    return value


def _json_object(response: httpx.Response) -> JsonObject:
    loaded = response.json()
    if not isinstance(loaded, dict):
        raise CloudflareClientError("Cloudflare API response was not a JSON object")
    return cast(JsonObject, loaded)


def _string_field(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise CloudflareClientError(f"Cloudflare API response missing string field: {key}")
    return value


def _etag(response: httpx.Response) -> str:
    value = response.headers.get("ETag") or response.headers.get("etag")
    if value is None or value == "":
        raise CloudflareClientError("R2 upload part response did not include an ETag header")
    return cast(str, value)


class CloudflareManifoldClient:
    """HTTP client for the prism-manifold-worker Zero-Trust API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        access_client_id: str | None = None,
        access_client_secret: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = (base_url or _env_required("PRISM_MANIFOLD_WORKER_URL")).rstrip("/")
        self._access_headers = {
            "CF-Access-Client-Id": access_client_id or _env_required("CF_ACCESS_CLIENT_ID"),
            "CF-Access-Client-Secret": access_client_secret or _env_required("CF_ACCESS_CLIENT_SECRET"),
        }
        self._http = httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _post_json(self, path: str, payload: JsonObject) -> JsonObject:
        response = await self._http.post(self._url(path), json=payload, headers=self._access_headers)
        response.raise_for_status()
        return _json_object(response)

    async def _upload_part(
        self,
        *,
        key: str,
        upload_id: str,
        part: UploadPart,
        expires_in_seconds: int,
    ) -> CompletedPart:
        signed = await self._post_json(
            "/api/v1/tensors/multipart/sign-part",
            {
                "key": key,
                "upload_id": upload_id,
                "part_number": part.part_number,
                "expires_in_seconds": expires_in_seconds,
            },
        )
        signed_url = _string_field(signed, "url")
        response = await self._http.put(
            signed_url,
            content=part.data,
            headers={"content-length": str(len(part.data))},
        )
        response.raise_for_status()
        return {"part_number": part.part_number, "etag": _etag(response)}

    async def upload_tensor_multipart(
        self,
        tensor_path: Path,
        *,
        key: str | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        concurrency: int = DEFAULT_CONCURRENCY,
        presign_expires_in_seconds: int = 900,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> MultipartUploadResult:
        """Upload a tensor artifact through concurrent R2 multipart upload parts."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)

        object_key = key or tensor_path.name
        create_payload: JsonObject = {
            "key": object_key,
            "content_type": content_type,
            "metadata": dict(metadata or {}),
        }
        create_response = await self._post_json("/api/v1/tensors/multipart/create", create_payload)
        upload_id = _string_field(create_response, "upload_id")

        queue: asyncio.Queue[UploadPart | None] = asyncio.Queue(maxsize=concurrency * 2)
        semaphore = asyncio.Semaphore(concurrency)
        completed_parts: list[CompletedPart] = []

        async def producer() -> None:
            part_number = 1
            with tensor_path.open("rb") as handle:
                while True:
                    chunk = await asyncio.to_thread(handle.read, chunk_size)
                    if not chunk:
                        break
                    await queue.put(UploadPart(part_number=part_number, data=chunk))
                    part_number += 1
            for _ in range(concurrency):
                await queue.put(None)

        async def upload_worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    async with semaphore:
                        completed = await self._upload_part(
                            key=object_key,
                            upload_id=upload_id,
                            part=item,
                            expires_in_seconds=presign_expires_in_seconds,
                        )
                    completed_parts.append(completed)
                finally:
                    queue.task_done()

        await asyncio.gather(producer(), *(upload_worker() for _ in range(concurrency)))
        ordered_parts = sorted(completed_parts, key=lambda item: item["part_number"])
        parts_payload: list[JsonValue] = [
            {"part_number": part["part_number"], "etag": part["etag"]} for part in ordered_parts
        ]
        complete_response = await self._post_json(
            "/api/v1/tensors/multipart/complete",
            {
                "key": object_key,
                "upload_id": upload_id,
                "parts": parts_payload,
            },
        )
        return {
            "key": object_key,
            "upload_id": upload_id,
            "part_count": len(ordered_parts),
            "parts": ordered_parts,
            "worker_response": complete_response,
        }

    async def query_nearest_scaffolds(self, latent_vector: Sequence[float], k: int = 10) -> JsonObject:
        return await self._post_json(
            "/api/v1/vectorize/query",
            {"latent_vector": [float(value) for value in latent_vector], "k": k},
        )

    async def commit_scaffold_update(
        self,
        scaffold_id: str,
        expected_version: int,
        data: Mapping[str, JsonValue],
    ) -> JsonObject:
        return await self._post_json(
            "/api/v1/scaffold/update",
            {
                "scaffold_id": scaffold_id,
                "expected_version": expected_version,
                "data": dict(data),
            },
        )


__all__ = ["CloudflareClientError", "CloudflareManifoldClient", "CompletedPart", "MultipartUploadResult"]
