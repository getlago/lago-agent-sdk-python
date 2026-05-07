"""boto3 bedrock-runtime wrapper.

Wraps `.converse`, `.converse_stream`, `.invoke_model`, `.invoke_model_with_response_stream`
in place — instrumentation never breaks the customer's call.

Critically: `invoke_model` returns a single-use streaming body. We consume it
once, parse JSON, extract usage, and **re-wrap** it as a fresh `StreamingBody`
so customer code reading `response['body'].read()` works unchanged.
"""
from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

from ..adapters import extract_bedrock_converse, extract_bedrock_invoke

logger = logging.getLogger("lago_agent_sdk.wrappers.boto3_bedrock")

_INSTRUMENTED_ATTR = "_lago_instrumented"
_LAGO_KWARG = "extra_lago"


def _pop_lago_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs.pop(_LAGO_KWARG, {}) or {}


def _restream_body(body_bytes: bytes) -> Any:
    """Build a botocore StreamingBody equivalent so customer .read() still works."""
    try:
        from botocore.response import StreamingBody
        return StreamingBody(io.BytesIO(body_bytes), len(body_bytes))
    except Exception:  # noqa: BLE001
        return io.BytesIO(body_bytes)


def wrap_boto3_bedrock_client(
    sdk: Any,
    client: Any,
    dimensions: dict[str, Any] | None = None,
    subscription: str | None = None,
) -> Any:
    """In-place wrap of a boto3 bedrock-runtime client. Idempotent."""
    if getattr(client, _INSTRUMENTED_ATTR, False):
        logger.info("lago: client already wrapped — skipping")
        return client

    base_dims = dict(dimensions or {})
    base_sub = subscription

    original_converse = client.converse
    original_converse_stream = getattr(client, "converse_stream", None)
    original_invoke_model = client.invoke_model
    original_invoke_stream = getattr(client, "invoke_model_with_response_stream", None)

    # ------------------------------------------------------------------
    # converse — non-streaming
    # ------------------------------------------------------------------
    def _converse(*args: Any, **kwargs: Any) -> Any:
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("modelId", "")
        try:
            response = original_converse(*args, **kwargs)
        except Exception:
            raise
        try:
            usage = extract_bedrock_converse(response, model_id=model_id)
            sdk.emit(
                usage,
                subscription=lago_opts.get("subscription") or base_sub,
                dimensions={**base_dims, **(lago_opts.get("dimensions") or {})},
            )
        except Exception as exc:  # noqa: BLE001 — never break the call
            logger.warning("lago: converse instrumentation failed: %s", exc)
        return response

    # ------------------------------------------------------------------
    # converse_stream — capture usage from final metadata event
    # ------------------------------------------------------------------
    def _converse_stream(*args: Any, **kwargs: Any) -> Any:
        assert original_converse_stream is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("modelId", "")
        response = original_converse_stream(*args, **kwargs)

        original_stream = response.get("stream")
        if original_stream is None:
            return response

        def _wrap_stream() -> Iterator[Any]:
            captured_usage: dict[str, Any] | None = None
            try:
                for event in original_stream:
                    if isinstance(event, dict) and "metadata" in event:
                        meta = event.get("metadata") or {}
                        if isinstance(meta, dict) and meta.get("usage"):
                            captured_usage = {"usage": meta["usage"]}
                    yield event
            finally:
                if captured_usage is not None:
                    try:
                        usage = extract_bedrock_converse(captured_usage, model_id=model_id)
                        sdk.emit(
                            usage,
                            subscription=lago_opts.get("subscription") or base_sub,
                            dimensions={**base_dims, **(lago_opts.get("dimensions") or {})},
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("lago: converse_stream instrumentation failed: %s", exc)

        response["stream"] = _wrap_stream()
        return response

    # ------------------------------------------------------------------
    # invoke_model — non-streaming. Consume body, parse, re-stream.
    # ------------------------------------------------------------------
    def _invoke_model(*args: Any, **kwargs: Any) -> Any:
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("modelId", "")
        response = original_invoke_model(*args, **kwargs)

        try:
            body = response.get("body")
            if body is not None:
                raw = body.read()
                response["body"] = _restream_body(raw)
                try:
                    parsed = json.loads(raw.decode("utf-8")) if raw else {}
                    usage = extract_bedrock_invoke(parsed, model_id=model_id)
                    sdk.emit(
                        usage,
                        subscription=lago_opts.get("subscription") or base_sub,
                        dimensions={**base_dims, **(lago_opts.get("dimensions") or {})},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("lago: invoke_model parse/emit failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — never break the call
            logger.warning("lago: invoke_model instrumentation failed: %s", exc)

        return response

    # ------------------------------------------------------------------
    # invoke_model_with_response_stream — capture usage from final metadata chunk
    # ------------------------------------------------------------------
    def _invoke_model_stream(*args: Any, **kwargs: Any) -> Any:
        assert original_invoke_stream is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("modelId", "")
        response = original_invoke_stream(*args, **kwargs)

        original_body = response.get("body")
        if original_body is None:
            return response

        def _wrap_invoke_stream() -> Iterator[Any]:
            # Two parallel sources: the model's own `usage` payload (Anthropic shape
            # or OpenAI-compat shape on the final delta), and Bedrock's invocation
            # metrics on the very last `message_stop` chunk. We accumulate both
            # without overwriting and pick the richer one at the end.
            usage_payload: dict[str, Any] = {}        # Anthropic / OpenAI-shape
            bedrock_metrics: dict[str, Any] = {}      # amazon-bedrock-invocationMetrics
            try:
                for event in original_body:
                    if isinstance(event, dict) and "chunk" in event:
                        chunk = event["chunk"]
                        chunk_bytes = chunk.get("bytes") if isinstance(chunk, dict) else None
                        if chunk_bytes:
                            try:
                                parsed = json.loads(chunk_bytes.decode("utf-8"))
                                if isinstance(parsed, dict):
                                    if isinstance(parsed.get("usage"), dict):
                                        # Merge — later chunks (message_delta) carry
                                        # the final output_tokens; earlier ones the
                                        # input_tokens.
                                        usage_payload = {**usage_payload, **parsed["usage"]}
                                    if isinstance(parsed.get("amazon-bedrock-invocationMetrics"), dict):
                                        bedrock_metrics = parsed["amazon-bedrock-invocationMetrics"]
                            except Exception:  # noqa: BLE001
                                pass
                    yield event
            finally:
                try:
                    # Prefer the model's own usage payload — it's the richer shape
                    # the existing adapter dispatch already understands. Fall back
                    # to Bedrock's metrics for models that don't emit one.
                    if usage_payload:
                        synthetic = {"usage": usage_payload}
                    elif bedrock_metrics:
                        synthetic = {
                            "usage": {
                                "prompt_tokens": bedrock_metrics.get("inputTokenCount", 0),
                                "completion_tokens": bedrock_metrics.get("outputTokenCount", 0),
                            }
                        }
                    else:
                        return
                    usage = extract_bedrock_invoke(synthetic, model_id=model_id)
                    sdk.emit(
                        usage,
                        subscription=lago_opts.get("subscription") or base_sub,
                        dimensions={**base_dims, **(lago_opts.get("dimensions") or {})},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("lago: invoke_model_with_response_stream instrumentation failed: %s", exc)

        response["body"] = _wrap_invoke_stream()
        return response

    client.converse = _converse
    if original_converse_stream is not None:
        client.converse_stream = _converse_stream
    client.invoke_model = _invoke_model
    if original_invoke_stream is not None:
        client.invoke_model_with_response_stream = _invoke_model_stream

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client
