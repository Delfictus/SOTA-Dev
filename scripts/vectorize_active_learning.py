from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import httpx


DEFAULT_WORKER_URL = "https://prism-manifold-worker.is-0b9.workers.dev"
DEFAULT_CREDENTIALS_ENV = Path.home() / ".config/prism/credentials.env"
DEFAULT_VECTORIZE_INDEX = "dkl_latent_space"


class AsyncPostClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        ...


class AsyncContextPostClient(AsyncPostClient, Protocol):
    async def __aenter__(self) -> Self:
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        ...


def cloudflare_headers(
    access_client_id: str | None = None,
    access_client_secret: str | None = None,
) -> dict[str, str]:
    load_prism_credentials_env()
    client_id = access_client_id or os.environ.get("CF_ACCESS_CLIENT_ID")
    client_secret = access_client_secret or os.environ.get("CF_ACCESS_CLIENT_SECRET")
    if not client_id or not client_secret:
        return {}
    return {
        "CF-Access-Client-Id": client_id,
        "CF-Access-Client-Secret": client_secret,
    }


def load_prism_credentials_env(path: Path | None = None) -> dict[str, str]:
    """Load local PRISM credential environment variables without printing secrets."""

    credentials_path = path or Path(os.environ.get("PRISM_CREDENTIALS_ENV", str(DEFAULT_CREDENTIALS_ENV)))
    if not credentials_path.is_file():
        return {}
    loaded: dict[str, str] = {}
    bare_token_count = 0
    for raw_line in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            # Some local credential vaults contain a standalone Cloudflare token
            # line. Never print or persist it; bind it to an explicit runtime key.
            if len(line) >= 40:
                bare_token_count += 1
                token_key = "CLOUDFLARE_VECTORIZE_API_TOKEN"
                if token_key not in os.environ:
                    os.environ[token_key] = line
                loaded[token_key] = line
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            values = shlex.split(raw_value, posix=True)
        except ValueError:
            values = [raw_value.strip()]
        value = values[0] if values else ""
        if key not in os.environ and value:
            os.environ[key] = value
        if value:
            loaded[key] = value
    return loaded


@asynccontextmanager
async def _client_context(client: AsyncPostClient | None) -> AsyncIterator[AsyncPostClient]:
    if client is not None:
        yield client
        return

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        yield http_client


def _json_from_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Vectorize worker response must be a JSON object")
    return payload


def _cloudflare_api_auth() -> tuple[str, str] | None:
    load_prism_credentials_env()
    account_id = os.environ.get("CF_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = (
        os.environ.get("CLOUDFLARE_VECTORIZE_API_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or os.environ.get("CF_API_TOKEN")
        or os.environ.get("CF_USER_API_TOKEN")
        or os.environ.get("CF_R2_CATALOG_TOKEN")
    )
    if not account_id or not token:
        return None
    return account_id, token


def _vectorize_index_name(index_name: str | None = None) -> str:
    return (
        index_name
        or os.environ.get("VECTORIZE_INDEX_NAME")
        or os.environ.get("PRISM_VECTORIZE_INDEX")
        or os.environ.get("VECTORIZE_INDEX")
        or DEFAULT_VECTORIZE_INDEX
    )


def _normalize_vectorize_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result")
    source = result if isinstance(result, dict) else payload
    matches = source.get("matches", []) if isinstance(source, dict) else []
    if not isinstance(matches, list):
        raise TypeError("Vectorize payload field 'matches' must be a list")
    return [match for match in matches if isinstance(match, dict)]


async def query_vectorize_neighbors(
    embedding: list[float],
    n: int = 20,
    worker_url: str = DEFAULT_WORKER_URL,
    index_name: str | None = None,
    access_client_id: str | None = None,
    access_client_secret: str | None = None,
    client: AsyncPostClient | None = None,
) -> list[dict[str, Any]]:
    """Query Cloudflare Vectorize for nearest thermodynamic neighbors."""
    headers = cloudflare_headers(access_client_id, access_client_secret)
    api_auth = _cloudflare_api_auth()
    if client is None and not headers and api_auth is not None:
        account_id, token = api_auth
        resolved_index = _vectorize_index_name(index_name)
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                "https://api.cloudflare.com/client/v4/"
                f"accounts/{account_id}/vectorize/v2/indexes/{resolved_index}/query",
                json={
                    "vector": embedding,
                    "topK": n,
                    "returnMetadata": "all",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
        payload = _json_from_response(response)
        return _normalize_vectorize_matches(payload)

    async with _client_context(client) as active_client:
        response = await active_client.post(
            f"{worker_url.rstrip('/')}/api/vectorize/query",
            json={
                "vector": embedding,
                "topK": n,
                "returnMetadata": True,
            },
            headers=headers or None,
        )
    payload = _json_from_response(response)
    return _normalize_vectorize_matches(payload)


async def update_d1_bald_priority(
    smiles: str,
    bald_adjustment: float,
    worker_url: str = DEFAULT_WORKER_URL,
    access_client_id: str | None = None,
    access_client_secret: str | None = None,
    client: AsyncPostClient | None = None,
) -> dict[str, Any]:
    """Persist a BALD priority adjustment through the edge worker."""
    headers = cloudflare_headers(access_client_id, access_client_secret)
    async with _client_context(client) as active_client:
        response = await active_client.post(
            f"{worker_url.rstrip('/')}/api/d1/bald-priority",
            json={
                "smiles": smiles,
                "bald_adjustment": bald_adjustment,
                "provenance": "vectorize_active_learning_v1",
            },
            headers=headers or None,
        )
    return _json_from_response(response)


async def update_bald_rankings(
    validated_smiles: str,
    validation_result: str,
    validated_embedding: list[float],
    n: int = 20,
    worker_url: str = DEFAULT_WORKER_URL,
    index_name: str | None = None,
    access_client_id: str | None = None,
    access_client_secret: str | None = None,
    client: AsyncPostClient | None = None,
) -> list[dict[str, Any]]:
    """Re-rank neighbors after a GPU MD validation result returns."""
    normalized = validation_result.upper()
    if normalized not in {"CONFIRMED", "REFUTED"}:
        raise ValueError("validation_result must be CONFIRMED or REFUTED")

    neighbors = await query_vectorize_neighbors(
        validated_embedding,
        n=n,
        worker_url=worker_url,
        index_name=index_name,
        access_client_id=access_client_id,
        access_client_secret=access_client_secret,
        client=client,
    )

    updates: list[dict[str, Any]] = []
    for neighbor in neighbors:
        metadata = neighbor.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        smiles = metadata.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            continue
        similarity = float(neighbor.get("score", 0.0))
        sign = 1.0 if normalized == "CONFIRMED" else -1.0
        bald_adjustment = sign * 0.5 * similarity
        response = await update_d1_bald_priority(
            smiles,
            bald_adjustment,
            worker_url=worker_url,
            access_client_id=access_client_id,
            access_client_secret=access_client_secret,
            client=client,
        )
        updates.append(
            {
                "validated_smiles": validated_smiles,
                "neighbor_smiles": smiles,
                "similarity": similarity,
                "bald_adjustment": bald_adjustment,
                "update_response": response,
            }
        )
    return updates


def _parse_embedding(value: str) -> list[float]:
    path = Path(value)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(value)
    if not isinstance(raw, list):
        raise TypeError("embedding must be a JSON list")
    return [float(item) for item in raw]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Vectorize and update BALD priorities.")
    parser.add_argument("--validated-smiles", required=True)
    parser.add_argument("--validation-result", choices=["CONFIRMED", "REFUTED"], required=True)
    parser.add_argument("--embedding", required=True, help="JSON list or path to a JSON list.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--worker-url", default=DEFAULT_WORKER_URL)
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--output", type=Path)
    return parser


async def async_main() -> None:
    args = build_parser().parse_args()
    updates = await update_bald_rankings(
        validated_smiles=args.validated_smiles,
        validation_result=args.validation_result,
        validated_embedding=_parse_embedding(args.embedding),
        n=args.top_k,
        worker_url=args.worker_url,
        index_name=args.index_name,
    )
    payload = {
        "validated_smiles": args.validated_smiles,
        "validation_result": args.validation_result,
        "neighbors_updated": len(updates),
        "updates": updates,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
