"""LLMod/LiteLLM cost tracking helpers.

These helpers read LLMod's LiteLLM proxy accounting endpoints. They do not call
models; they only fetch spend metadata for already-made requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from urllib.request import Request, urlopen

from src.config import Config, load_config


@dataclass(frozen=True)
class SpendSnapshot:
    """Current spend/budget state for the configured LLMod key/user."""

    taken_at: datetime
    key_spend: Decimal
    user_spend: Decimal
    max_budget: Decimal | None


@dataclass(frozen=True)
class SpendLog:
    """One row from `/spend/logs`, sanitized for local accounting."""

    request_id: str
    call_type: str
    model_group: str
    model: str
    spend: Decimal
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    start_time: datetime
    end_time: datetime | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def llmod_base_url(cfg: Config) -> str:
    """Return the LLMod proxy base without the OpenAI `/v1` suffix."""
    return cfg.llmod_base_url.rstrip("/").removesuffix("/v1")


def _decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def parse_llmod_time(value: str | None) -> datetime | None:
    """Parse LLMod ISO timestamps, normalizing trailing Z to UTC."""
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def _get_json(path: str, cfg: Config) -> Any:
    url = f"{llmod_base_url(cfg)}{path}"
    req = Request(url, headers={"Authorization": f"Bearer {cfg.llmod_api_key}"})
    with urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed HTTPS API URL from config
        return json.loads(resp.read().decode("utf-8"))


def parse_spend_snapshot(
    key_info: dict[str, Any],
    user_info: dict[str, Any],
    *,
    taken_at: datetime | None = None,
) -> SpendSnapshot:
    info = key_info.get("info") or {}
    max_budget = user_info.get("max_budget")
    return SpendSnapshot(
        taken_at=taken_at or utc_now(),
        key_spend=_decimal(info.get("spend")),
        user_spend=_decimal(user_info.get("spend")),
        max_budget=None if max_budget is None else _decimal(max_budget),
    )


def get_spend_snapshot(cfg: Config | None = None) -> SpendSnapshot:
    if cfg is None:
        cfg = load_config()
    taken_at = utc_now()
    key_info = _get_json("/key/info", cfg)
    user_info = _get_json("/v2/user/info", cfg)
    return parse_spend_snapshot(key_info, user_info, taken_at=taken_at)


def parse_spend_log(row: dict[str, Any]) -> SpendLog:
    start = parse_llmod_time(row.get("startTime"))
    if start is None:
        raise ValueError("spend log row missing startTime")
    return SpendLog(
        request_id=str(row.get("request_id") or ""),
        call_type=str(row.get("call_type") or ""),
        model_group=str(row.get("model_group") or ""),
        model=str(row.get("model") or ""),
        spend=_decimal(row.get("spend")),
        total_tokens=int(row.get("total_tokens") or 0),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        start_time=start,
        end_time=parse_llmod_time(row.get("endTime")),
    )


def parse_spend_logs(payload: Any) -> list[SpendLog]:
    rows = payload if isinstance(payload, list) else []
    out: list[SpendLog] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(parse_spend_log(row))
    return out


def get_spend_logs(cfg: Config | None = None) -> list[SpendLog]:
    if cfg is None:
        cfg = load_config()
    return parse_spend_logs(_get_json("/spend/logs", cfg))


def filter_logs(
    logs: list[SpendLog],
    *,
    start: datetime,
    end: datetime,
) -> list[SpendLog]:
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    return [row for row in logs if start <= row.start_time <= end]


def summarize_logs(logs: list[SpendLog]) -> dict[str, Any]:
    """Group logs by call_type/model_group with Decimal-safe string totals."""
    total_spend = sum((row.spend for row in logs), Decimal("0"))
    total_tokens = sum(row.total_tokens for row in logs)
    total_prompt = sum(row.prompt_tokens for row in logs)
    total_completion = sum(row.completion_tokens for row in logs)
    groups: dict[str, dict[str, Any]] = {}
    for row in logs:
        key = f"{row.call_type}|{row.model_group}"
        group = groups.setdefault(
            key,
            {
                "call_type": row.call_type,
                "model_group": row.model_group,
                "count": 0,
                "spend_usd": "0",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )
        group["count"] += 1
        group["spend_usd"] = str(_decimal(group["spend_usd"]) + row.spend)
        group["total_tokens"] += row.total_tokens
        group["prompt_tokens"] += row.prompt_tokens
        group["completion_tokens"] += row.completion_tokens
    return {
        "count": len(logs),
        "spend_usd": str(total_spend),
        "total_tokens": total_tokens,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "by_call_type_model": list(groups.values()),
    }


def public_config_snapshot(cfg: Config) -> dict[str, Any]:
    """Config snapshot for ledgers with all secrets omitted."""
    return {
        "chunk_size": cfg.chunk_size,
        "overlap_ratio": cfg.overlap_ratio,
        "top_k": cfg.top_k,
        "retrieval_fetch_k": cfg.retrieval_fetch_k,
        "embed_content": cfg.embed_content,
        "embed_model": cfg.embed_model,
        "embed_dim": cfg.embed_dim,
        "chat_model": cfg.chat_model,
        "reasoning_effort": cfg.reasoning_effort,
        "pinecone_index": cfg.pinecone_index,
        "pinecone_namespace": cfg.pinecone_namespace,
        "llmod_base_url": cfg.llmod_base_url,
    }


def assert_no_secrets(obj: Any, cfg: Config) -> None:
    """Raise if a ledger object accidentally contains raw API keys."""
    text = json.dumps(obj, ensure_ascii=True, sort_keys=True)
    for secret in (cfg.llmod_api_key, cfg.pinecone_api_key):
        if secret and secret in text:
            raise ValueError("ledger row contains a raw secret")
