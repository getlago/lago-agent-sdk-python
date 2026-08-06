"""wrap()-triggered automatic, non-blocking pricing warm-up.

Covers `LagoSDK._auto_prime_pricing_for`/`_extract_mistral_api_key`: the
customer calls `sdk.wrap(client)` (already part of their normal flow, no new
function to remember) and that alone should be enough for the session's
FIRST Mistral/Workers AI call to have a real shot at pricing correctly,
without ever declaring `LagoConfig.mistral_api_key` separately — the client
being wrapped already carries the exact credential needed.
"""

from __future__ import annotations

import time
from decimal import Decimal

from lago_agent_sdk import LagoConfig, LagoSDK, ModelPrice
from lago_agent_sdk.pricing import HttpPricingFetcher, PricingProvider, parse_mistral_aliases


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """`LagoSDK.wrap()` wakes the REAL background queue thread (see
    `EventQueue.wake()`), which races any direct `provider.maybe_refresh()`
    call in the test's own thread — both are legitimate, concurrent
    triggers. Poll instead of asserting immediately after one call."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeSecurity:
    def __init__(self, api_key: str):
        self.api_key = api_key


class _FakeSdkConfiguration:
    def __init__(self, api_key: str):
        self.security = _FakeSecurity(api_key)


class FakeMistralClient:
    """Mimics the real shape verified against mistralai.client.Mistral:
    `client.sdk_configuration.security.api_key`."""

    __module__ = "mistralai.client.sdk"

    def __init__(self, api_key: str):
        self.sdk_configuration = _FakeSdkConfiguration(api_key)


class FakeOpenAIClient:
    """Mimics openai.OpenAI's `base_url` attribute (a plain str is enough —
    real usage is an httpx.URL, but only `str(...)` on it is ever read)."""

    __module__ = "openai.client"

    def __init__(self, base_url: str):
        self.base_url = base_url


_MISTRAL_ALIASES = parse_mistral_aliases(
    {"data": [{"id": "mistral-small-2603", "aliases": ["mistral-small-latest"]}]}
)
_OPENROUTER = {
    "exact": {},
    "norm": {
        # matches parse_openrouter's own convention: (vendor, full suffix after "vendor/")
        ("mistralai", "mistral-small-2603"): ModelPrice(
            source="openrouter", input=Decimal("0.00000015"), output=Decimal("0.0000006")
        )
    },
}


class _CloudflareCallCountingFetcher(HttpPricingFetcher):
    """Shared by both wrap()-vs-Cloudflare-base_url tests below."""

    def __init__(self):
        super().__init__()
        self.cloudflare_calls = 0

    def fetch_cloudflare_workers_ai(self):
        self.cloudflare_calls += 1
        return {}


def _sdk_with_provider(provider: PricingProvider) -> LagoSDK:
    cfg = LagoConfig(
        api_key="dummy", default_subscription_id="sub_test", pricing_mode="price", pricing_provider=provider
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]
    return sdk


def test_extract_mistral_api_key_reads_the_real_attribute_path():
    client = FakeMistralClient(api_key="sk-from-client")
    assert LagoSDK._extract_mistral_api_key(client) == "sk-from-client"


def test_extract_mistral_api_key_returns_none_when_attribute_missing():
    class Empty:
        pass

    assert LagoSDK._extract_mistral_api_key(Empty()) is None


def test_wrap_mistral_learns_key_and_primes_without_config_key():
    """The whole point: no LagoConfig.mistral_api_key anywhere, and the
    session's first Mistral lookup still resolves correctly because wrap()
    learned the key from the client and kicked off the fetch."""

    class _StubFetcher(HttpPricingFetcher):
        def __init__(self):
            super().__init__()
            self.seen_keys: list[str | None] = []

        def fetch_mistral_aliases(self, api_key=None):
            self.seen_keys.append(api_key)
            return _MISTRAL_ALIASES

        def fetch_openrouter(self):
            return _OPENROUTER

    fetcher = _StubFetcher()
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    sdk = _sdk_with_provider(provider)

    client = FakeMistralClient(api_key="sk-from-client")
    sdk.wrap(client)  # <-- the only thing the customer does

    assert _wait_until(lambda: fetcher.seen_keys == ["sk-from-client"])
    mp = provider.lookup("mistral", "mistral-small-latest", "native")
    assert mp is not None
    assert mp.input == Decimal("0.00000015")


def test_wrap_openai_pointed_at_cloudflare_gateway_primes_workers_ai():
    fetcher = _CloudflareCallCountingFetcher()
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    sdk = _sdk_with_provider(provider)

    client = FakeOpenAIClient(base_url="https://gateway.ai.cloudflare.com/v1/acct/gw/compat")
    sdk.wrap(client)

    assert _wait_until(lambda: fetcher.cloudflare_calls == 1)


def test_wrap_openai_pointed_at_real_openai_does_not_prime_workers_ai():
    """A generic OpenAI client NOT pointed at Cloudflare must not trigger the
    Workers AI fetch — only the base_url signal should do that."""
    fetcher = _CloudflareCallCountingFetcher()
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    sdk = _sdk_with_provider(provider)

    client = FakeOpenAIClient(base_url="https://api.openai.com/v1")
    sdk.wrap(client)
    provider.maybe_refresh()

    assert fetcher.cloudflare_calls == 0


def test_auto_prime_is_a_noop_in_token_mode():
    """No point flagging anything stale for a customer who never opted into
    price mode — the credential-gated sources should stay completely untouched."""

    class _StubFetcher(HttpPricingFetcher):
        def __init__(self):
            super().__init__()
            self.mistral_calls = 0

        def fetch_mistral_aliases(self, api_key=None):
            self.mistral_calls += 1
            return {}

    fetcher = _StubFetcher()
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    cfg = LagoConfig(
        api_key="dummy", default_subscription_id="sub_test", pricing_provider=provider
    )  # tokens (default)
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]

    sdk.wrap(FakeMistralClient(api_key="sk-from-client"))
    provider.maybe_refresh()

    assert fetcher.mistral_calls == 0
