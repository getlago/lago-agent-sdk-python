# Contributing

## Development setup

```bash
git clone https://github.com/getlago/lago-agent-sdk-python
cd lago-agent-sdk-python
python3.11 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

## Run tests

```bash
# Unit tests (fast, no network)
pytest tests/unit

# Integration tests (require credentials — see env vars in each test)
AWS_BEARER_TOKEN_BEDROCK="..." \
MISTRAL_API_KEY="..." \
LAGO_API_URL="..." LAGO_API_KEY="..." LAGO_EXTERNAL_SUBSCRIPTION_ID="..." \
pytest tests/integration
```

## Linting and type checks

```bash
ruff check src tests
ruff format --check src tests
mypy src
```

## Where things live

- `src/lago_agent_sdk/` — the SDK
- `src/lago_agent_sdk/adapters/` — one file per (provider, access path); transforms provider responses into `CanonicalUsage`
- `src/lago_agent_sdk/wrappers/` — one file per (provider SDK, access path); patches client objects in place
- `src/lago_agent_sdk/canonical.py` — the normalized usage shape sent to Lago
- `src/lago_agent_sdk/queue.py` — async event queue with backoff
- `src/lago_agent_sdk/lago_client.py` — thin HTTP client to `/events/batch`
- `tests/unit/` — unit tests, organized to mirror `src/`
- `tests/unit/adapters/fixtures/` — captured real provider responses, used by adapter tests
- `tests/integration/` — live tests, gated on credential env vars

## Adding a provider

1. Capture real fixtures: write a small script that hits the provider and saves responses to `tests/unit/adapters/fixtures/<provider>/`.
2. Write the adapter at `src/lago_agent_sdk/adapters/<provider>.py` that returns `CanonicalUsage`.
3. Write the wrapper at `src/lago_agent_sdk/wrappers/<provider>.py` that intercepts the customer-facing method.
4. Update `detector.py` to recognize the client class.
5. Update `sdk.py::wrap()` to dispatch to the new wrapper.
6. Add unit tests against the captured fixtures.
7. Add a live integration test gated on the provider's API key env var.

## Pull request checklist

- [ ] Unit tests cover the change
- [ ] Existing tests still pass
- [ ] Linter clean (`ruff check`, `mypy src`)
- [ ] CHANGELOG.md updated under `## [Unreleased]`
- [ ] Doc updated if public API changed
