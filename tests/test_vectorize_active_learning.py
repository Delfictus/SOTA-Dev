from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from scripts.vectorize_active_learning import (
    load_prism_credentials_env,
    query_vectorize_neighbors,
    update_bald_rankings,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        if url.endswith("/api/vectorize/query"):
            return FakeResponse(
                {
                    "matches": [
                        {"score": 0.9, "metadata": {"smiles": "CCO"}},
                        {"score": 0.4, "metadata": {"smiles": "CCC"}},
                    ]
                }
            )
        return FakeResponse({"ok": True})


def test_query_vectorize_neighbors_uses_mock_client_without_credentials() -> None:
    async def run() -> None:
        client = FakeClient()
        matches = await query_vectorize_neighbors([0.1, 0.2], n=2, client=client)
        assert len(matches) == 2
        assert client.calls[0]["headers"] is None
        assert client.calls[0]["json"]["topK"] == 2
        assert client.calls[0]["json"]["returnMetadata"] is True

    asyncio.run(run())


def test_confirmed_validation_boosts_neighbor_bald_priority() -> None:
    async def run() -> None:
        client = FakeClient()
        updates = await update_bald_rankings(
            validated_smiles="NCC",
            validation_result="CONFIRMED",
            validated_embedding=[0.1, 0.2],
            n=2,
            client=client,
        )
        assert [u["neighbor_smiles"] for u in updates] == ["CCO", "CCC"]
        assert updates[0]["bald_adjustment"] == 0.45
        update_calls = [call for call in client.calls if call["url"].endswith("/api/d1/bald-priority")]
        assert len(update_calls) == 2
        assert update_calls[0]["json"]["smiles"] == "CCO"

    asyncio.run(run())


def test_refuted_validation_demotes_neighbor_bald_priority() -> None:
    async def run() -> None:
        client = FakeClient()
        updates = await update_bald_rankings(
            validated_smiles="NCC",
            validation_result="REFUTED",
            validated_embedding=[0.1, 0.2],
            n=2,
            client=client,
        )
        assert updates[0]["bald_adjustment"] == -0.45

    asyncio.run(run())


def test_load_prism_credentials_env_does_not_override_existing(tmp_path: Path, monkeypatch: Any) -> None:
    credentials = tmp_path / "credentials.env"
    credentials.write_text(
        "export CF_ACCOUNT_ID=account_from_file\n"
        "CLOUDFLARE_API_TOKEN='token_from_file'\n"
        "EXISTING_KEY=from_file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_KEY", "keep_existing")

    loaded = load_prism_credentials_env(credentials)

    assert loaded["CF_ACCOUNT_ID"] == "account_from_file"
    assert loaded["CLOUDFLARE_API_TOKEN"] == "token_from_file"
    assert "token_from_file" not in repr(loaded.keys())
    assert loaded["EXISTING_KEY"] == "from_file"
    assert os.environ["EXISTING_KEY"] == "keep_existing"


def test_load_prism_credentials_env_binds_bare_vectorize_token(tmp_path: Path, monkeypatch: Any) -> None:
    credentials = tmp_path / "credentials.env"
    bare_token = "x" * 53
    credentials.write_text(
        "CF_ACCOUNT_ID=account_from_file\n"
        f"{bare_token}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CLOUDFLARE_VECTORIZE_API_TOKEN", raising=False)

    loaded = load_prism_credentials_env(credentials)

    assert loaded["CLOUDFLARE_VECTORIZE_API_TOKEN"] == bare_token
    assert os.environ["CLOUDFLARE_VECTORIZE_API_TOKEN"] == bare_token
