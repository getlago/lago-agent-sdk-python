"""LagoSDK tests — emit, subscription resolution, error policy."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pytest

from lago_agent_sdk import CanonicalUsage, LagoConfig, LagoSDK
from lago_agent_sdk.exceptions import UnknownClientError


def _new_sdk(default_sub: str | None = None, sender=None) -> tuple[LagoSDK, list]:
    received: list = []
    sender = sender or (lambda b: received.append(list(b)))
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    return sdk, received


def test_emit_only_nonzero_fields():
    sdk, received = _new_sdk(default_sub="sub_default")
    u = CanonicalUsage(input=10, output=20, cache_read=0, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}
    for e in flat:
        assert e["external_subscription_id"] == "sub_default"


def test_per_call_subscription_overrides_contextvar_and_default():
    sdk, received = _new_sdk(default_sub="sub_default")
    tok = sdk.set_subscription("sub_ctx")
    try:
        u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
        sdk.emit(u, subscription="sub_call")
        assert sdk.flush(timeout=2.0)
        sdk.shutdown(timeout=1.0)
    finally:
        sdk.reset_subscription(tok)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_call" for e in flat)


def test_contextvar_overrides_default():
    sdk, received = _new_sdk(default_sub="sub_default")
    tok = sdk.set_subscription("sub_ctx")
    try:
        u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
        sdk.emit(u)
        assert sdk.flush(timeout=2.0)
        sdk.shutdown(timeout=1.0)
    finally:
        sdk.reset_subscription(tok)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_ctx" for e in flat)


def test_no_resolvable_subscription_drops_events():
    sdk, received = _new_sdk(default_sub=None)
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)
    assert sdk.flush(timeout=1.0)
    sdk.shutdown(timeout=1.0)
    assert not received  # nothing emitted


def test_emit_never_raises_on_inner_failure():
    """emit() must swallow internal errors — instrumentation never breaks the call."""
    sdk, _ = _new_sdk(default_sub="sub")
    # Force the queue to be broken
    sdk._queue.push = lambda e: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)  # must not raise
    sdk.shutdown(timeout=1.0)


def test_wrap_unknown_client_raises_at_wrap_time():
    sdk, _ = _new_sdk()
    with pytest.raises(UnknownClientError):
        sdk.wrap(object())
    sdk.shutdown(timeout=1.0)


# ----------------------------------------------------------------------
# Constructor precedence. `api_url`'s default used to be the production URL, so
# `if api_url:` always fired and clobbered a config-supplied one — sending a
# local-dev customer's events to production Lago.
# ----------------------------------------------------------------------
def test_config_only_api_url_survives():
    """The bug: a customer who configures ONLY via LagoConfig must not have their
    events redirected to production."""
    sdk = LagoSDK(api_key="k", config=LagoConfig(api_url="http://localhost:3000/api/v1"))
    try:
        assert sdk.config.api_url == "http://localhost:3000/api/v1"
    finally:
        sdk.shutdown(timeout=1.0)


def test_explicit_api_url_still_wins_over_config():
    """The documented rule — explicit args beat config — must still hold."""
    sdk = LagoSDK(
        api_key="k",
        api_url="http://explicit:3000/api/v1",
        config=LagoConfig(api_url="http://fromconfig:3000/api/v1"),
    )
    try:
        assert sdk.config.api_url == "http://explicit:3000/api/v1"
    finally:
        sdk.shutdown(timeout=1.0)


@pytest.mark.parametrize("empty", ["", None])
def test_empty_or_absent_api_url_keeps_the_production_default(empty):
    """`api_url=os.environ.get("LAGO_API_URL", "")` with the var unset must NOT write
    "". Downstream that is unrecoverable: requests raises MissingSchema, which is not a
    LagoApiError, so the queue treats it as transient and retries at the 60s ceiling
    forever — all billing stops with only a growing buffer as the symptom."""
    sdk = LagoSDK(api_key="k", api_url=empty)
    try:
        assert sdk.config.api_url == "https://api.getlago.com/api/v1"
    finally:
        sdk.shutdown(timeout=1.0)


def test_default_api_url_is_still_production_when_nothing_is_passed():
    """Changing the parameter default to None must not change this."""
    sdk = LagoSDK(api_key="k")
    try:
        assert sdk.config.api_url == "https://api.getlago.com/api/v1"
    finally:
        sdk.shutdown(timeout=1.0)


def test_verify_ssl_needs_no_config_object():
    """A local Lago on a self-signed cert is reachable without building a
    LagoConfig — which is what pushed callers toward the clobber in the first
    place, since a custom api_url and verify_ssl=False go together."""
    sdk = LagoSDK(api_key="k", api_url="https://api.lago.dev/api/v1", verify_ssl=False)
    try:
        assert sdk.config.verify_ssl is False
        assert sdk._lago_client.verify_ssl is False
    finally:
        sdk.shutdown(timeout=1.0)


def test_explicit_verify_ssl_wins_over_config():
    sdk = LagoSDK(api_key="k", verify_ssl=True, config=LagoConfig(verify_ssl=False))
    try:
        assert sdk.config.verify_ssl is True
    finally:
        sdk.shutdown(timeout=1.0)


def test_ignored_usd_cost_is_reported_not_silently_dropped():
    """A caller who supplies a real metered cost while the effective mode isn't
    'price' had it discarded with no log and no on_error — so a hand-rolled
    backfill could bill token counts only and look successful."""
    errors: list = []
    received: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub",
        pricing_mode="tokens",
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.extend(b)  # type: ignore[attr-defined]
    u = CanonicalUsage(input=10, output=5, model="m", provider="anthropic", api="native")
    sdk.emit(u, usd_cost=0.0123)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    assert errors, "an ignored usd_cost must reach on_error"
    msg, where = errors[0]
    assert "usd_cost" in msg and "0.0123" in msg
    assert where == "pricing"
    # And the call is still billed as token counts — reporting must not drop events.
    assert {e["code"] for e in received} == {"llm_input_tokens", "llm_output_tokens"}


def test_no_usd_cost_in_token_mode_reports_nothing():
    """The common case must stay silent — only an explicitly supplied cost that
    gets discarded is worth reporting."""
    errors: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub",
        pricing_mode="tokens",
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]
    sdk.emit(CanonicalUsage(input=10, output=5, model="m", provider="anthropic", api="native"))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert errors == []


def test_no_resolvable_subscription_is_reported_not_just_logged():
    """Dropping a call for lack of a subscription loses its billing entirely, so it
    must reach on_error — the documented channel for every other billing gap. It
    was logger.error only, while the JS port already reported it."""
    errors: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id=None,
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]
    sdk.emit(CanonicalUsage(input=10, model="m", provider="p", api="x"))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert errors, "a dropped call must reach on_error"
    assert "subscription" in errors[0][0]


def test_negative_counts_are_never_emitted():
    """`nonzero_numeric` filtered on truthiness, so a negative survived and was
    emitted verbatim as value="-100" — a negative billable quantity. JS already
    filtered on > 0."""
    sdk, received = _new_sdk(default_sub="sub")
    sdk.emit(CanonicalUsage(input=-100, output=5, model="m", provider="p", api="x"))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert {e["code"] for e in flat} == {"llm_output_tokens"}
    assert all(float(e["properties"]["value"]) > 0 for e in flat)


def test_dimensions_merge_into_event_properties():
    sdk, received = _new_sdk(default_sub="sub")
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u, dimensions={"project": "demo", "tenant": "acme"})
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert flat[0]["properties"]["project"] == "demo"
    assert flat[0]["properties"]["tenant"] == "acme"


def test_caller_dimensions_win_on_a_collision_on_both_emitters():
    """One rule across both paths: a caller dimension overrides every
    SDK-computed property of the same name.

    The cost path used to spread dimensions into `base_properties`, i.e. BEFORE
    `unit`/`value`/`base_cost`/`unit_price`, so those four silently overwrote a
    same-named caller dimension there while the token path honoured it. Same
    customer config, two different outcomes depending on the mode.
    """
    dims = {"unit": "seat", "value": "CUSTOM", "model": "my-label", "team": "platform"}
    u = CanonicalUsage(input=100, output=50, model="claude-sonnet-4-5", provider="anthropic", api="native")

    # Token path.
    sdk_tok, got_tok = _new_sdk(default_sub="sub")
    sdk_tok.emit(u, dimensions=dims, mode="tokens")
    assert sdk_tok.flush(timeout=2.0)
    sdk_tok.shutdown(timeout=1.0)
    tok = [e for batch in got_tok for e in batch]

    # Cost path (precomputed, so no price table needed).
    sdk_cost, got_cost = _new_sdk(default_sub="sub")
    sdk_cost.emit(u, dimensions=dims, mode="price", usd_cost=0.01)
    assert sdk_cost.flush(timeout=2.0)
    sdk_cost.shutdown(timeout=1.0)
    cost = [e for batch in got_cost for e in batch]

    assert tok and cost
    for label, events in (("token", tok), ("cost", cost)):
        for e in events:
            p = e["properties"]
            assert p["unit"] == "seat", f"{label}: caller `unit` must win"
            assert p["value"] == "CUSTOM", f"{label}: caller `value` must win"
            assert p["model"] == "my-label", f"{label}: caller `model` must win"
            assert p["team"] == "platform"

    # The accepted consequence of that rule, pinned deliberately: a dimension
    # named `value` overrides the reported quantity. It is NOT able to touch the
    # charged amount on a cost event, because `precise_total_amount_cents` is a
    # sibling of `properties`, not a member of it.
    assert cost[0]["precise_total_amount_cents"] == "1"


def test_negative_token_counts_are_reported_not_just_dropped(caplog) -> None:
    """`CanonicalUsage` is exported and `emit()` takes one directly — the documented
    way to backfill usage the SDK did not intercept — so a caller computing a delta
    wrongly really can hand us a negative. Dropping it was correct (Lago would sum a
    negative billable quantity) but it was the only drop path that never reached
    on_error."""
    seen: list[tuple[Exception, str]] = []
    received: list = []
    cfg = LagoConfig(api_key="k", default_subscription_id="sub", on_error=lambda e, c: seen.append((e, c)))
    sdk = LagoSDK(api_key="k", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    try:
        sdk.emit(CanonicalUsage(input=-100, output=50, model="m", provider="anthropic", api="native"))
        assert sdk.flush(timeout=2.0)
    finally:
        sdk.shutdown(timeout=1.0)

    flat = [e for batch in received for e in batch]
    values = [e["properties"]["value"] for e in flat]
    assert all(not str(v).startswith("-") for v in values), f"a negative was billed: {values}"
    assert "negative_tokens" in [c for _, c in seen], f"drop must reach on_error; got {seen}"
    assert "input" in str(next(e for e, c in seen if c == "negative_tokens"))


def test_a_dropped_event_logs_exactly_once(caplog) -> None:
    """`_report_error` invokes on_error AND logs. An extra logger.error alongside it
    emitted the same drop twice under two levels, so a customer grepping logs counted
    one lost call as two — while the JS port logged nothing at all for it."""
    cfg = LagoConfig(api_key="k")  # no default subscription -> the drop path
    sdk = LagoSDK(api_key="k", config=cfg)
    try:
        with caplog.at_level(logging.DEBUG, logger="lago_agent_sdk"):
            sdk.emit(CanonicalUsage(input=10, output=5, model="m", provider="anthropic", api="native"))
        drop_lines = [r for r in caplog.records if "subscription" in r.getMessage()]
        assert len(drop_lines) == 1, f"expected one line per drop, got {[r.getMessage() for r in drop_lines]}"
    finally:
        sdk.shutdown(timeout=1.0)


def test_verify_ssl_false_survives_a_broken_urllib3(monkeypatch) -> None:
    """Suppressing the InsecureRequestWarning is an optional convenience; it must
    never be able to fail construction. This sits on an advertised path — `verify_ssl`
    is a first-class constructor arg the docstring recommends for local dev — so an
    ImportError/AttributeError here would take down `LagoSDK()` for exactly the setup
    the flag exists to serve. The old code reached through `requests.packages`, a
    legacy shim with no guarantee of existing."""
    import builtins

    real_import = builtins.__import__

    def boom(name, *args, **kwargs):
        if name == "urllib3":
            raise ImportError("no urllib3 for you")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", boom)
    # Also remove the legacy shim the old code reached through, so this test fails
    # against that version instead of silently passing: `requests.packages` is a
    # compatibility alias, not API, and nothing guarantees it is present.
    import requests

    monkeypatch.delattr(requests, "packages", raising=False)

    sdk = LagoSDK(api_key="k", api_url="https://example.invalid/api/v1", verify_ssl=False)
    try:
        assert sdk.config.verify_ssl is False
    finally:
        sdk.shutdown(timeout=1.0)


# --------------------------------------------------------------------------
# Event time — a backfill must bill into the period the usage happened in
# --------------------------------------------------------------------------
def test_emit_stamps_the_given_instant_on_every_event_not_now() -> None:
    """Without this, a replay of last week's logs billed every call into the period
    the script happened to run in, and nothing in Lago could tell afterwards."""
    sdk, received = _new_sdk(default_sub="sub")
    when = datetime(2026, 8, 7, 14, 22, 3, tzinfo=timezone.utc)
    u = CanonicalUsage(input=10, output=20, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u, timestamp=when)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert len(flat) == 2
    # One instant for the whole call: Lago sums these into a period, so a call must
    # never straddle two of them because two `time.time()` reads disagreed.
    assert {e["timestamp"] for e in flat} == {int(when.timestamp())}


def test_emit_stamps_a_cost_event_too() -> None:
    """The cost path reads its own clock, so it needed threading separately from the
    token path — and a backfill of BYOK spend goes down this one."""
    sdk, received = _new_sdk(default_sub="sub")
    when = datetime(2026, 8, 7, 14, 0, 0, tzinfo=timezone.utc)
    u = CanonicalUsage(input=10, output=20, model="m", provider="anthropic", api="native")
    sdk.emit(u, mode="price", usd_cost=0.0011187, timestamp=when)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert [e["code"] for e in flat] == ["llm_cost"]
    assert flat[0]["timestamp"] == int(when.timestamp())


def test_emit_accepts_epoch_seconds() -> None:
    sdk, received = _new_sdk(default_sub="sub")
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u, timestamp=1786112523)
    # A float is what `datetime.timestamp()` hands back, so it must not be refused.
    sdk.emit(u, timestamp=1786112523.987)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert {e["timestamp"] for e in flat} == {1786112523}


def test_a_naive_timestamp_is_read_as_utc() -> None:
    """Same rule as `_interval_sql`'s window bound, and the same rule the JS port
    applies to a `Date` — otherwise a caller who reads a window and bills it has the
    two disagree by their machine's UTC offset."""
    sdk, received = _new_sdk(default_sub="sub")
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u, timestamp=datetime(2026, 8, 7, 14, 22, 3))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert flat[0]["timestamp"] == int(datetime(2026, 8, 7, 14, 22, 3, tzinfo=timezone.utc).timestamp())


def test_an_unreadable_timestamp_is_reported_and_still_bills() -> None:
    """Never silently under-bill: a bad timestamp is a reconciliation problem the
    operator can see and fix, while dropping the event is revenue that never appears.
    An ISO string is the likely mistake, and is deliberately not accepted."""
    errors: list = []
    received: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub",
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.extend(b)  # type: ignore[attr-defined]
    before = int(time.time())
    sdk.emit(CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke"), timestamp="2026-08-07Z")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    assert errors, "an unreadable timestamp must reach on_error"
    msg, where = errors[0]
    assert "2026-08-07Z" in msg and where == "timestamp"
    # ...and the call is still billed, at now.
    assert len(received) == 1
    assert before <= received[0]["timestamp"] <= int(time.time())


def test_a_numeric_string_is_refused_not_coerced() -> None:
    """`int("1786112523")` would sail through where the isinstance check rejects it —
    the same input must not bill in one repo and report an error in the other. The JS
    port's `Number()` is the one that would coerce, so this is pinned on both sides."""
    errors: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub",
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]
    sdk.emit(CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke"), timestamp="1786112523")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert [where for _, where in errors] == ["timestamp"]


def test_no_timestamp_still_stamps_now() -> None:
    """The live `wrap()` path passes nothing and must be unchanged by all of this."""
    sdk, received = _new_sdk(default_sub="sub")
    before = int(time.time())
    sdk.emit(CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke"))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert before <= flat[0]["timestamp"] <= int(time.time())
