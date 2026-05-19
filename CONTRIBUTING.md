# Contributing

## Development setup

Recommended: [uv](https://docs.astral.sh/uv/) (10× faster installs, lockfile-driven reproducible envs):

```bash
git clone https://github.com/getlago/lago-agent-sdk-python
cd lago-agent-sdk-python
uv sync --all-extras       # creates .venv, installs from uv.lock
```

Plain pip works too:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
```

Common workflows are wired through the Makefile:

```bash
make sync     # install/sync deps from uv.lock
make test     # unit tests
make lint     # ruff check + ruff format --check + mypy
make format   # auto-fix lint and format
make check    # lint + test (what CI runs)
```

## Run tests

```bash
# Unit tests (fast, no network)
make test

# Unit tests with coverage report
uv run pytest tests/unit --cov=lago_agent_sdk --cov-report=term-missing

# Integration tests (require credentials — see env vars in each test)
AWS_BEARER_TOKEN_BEDROCK="..." \
MISTRAL_API_KEY="..." \
LAGO_API_URL="..." LAGO_API_KEY="..." LAGO_EXTERNAL_SUBSCRIPTION_ID="..." \
uv run pytest tests/integration -q
```

## Linting and type checks

```bash
make lint        # all three at once
# or directly:
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
```

CI gates on all of the above plus an 80% coverage floor. Raising the floor is encouraged as coverage improves.

## Updating dependencies

```bash
uv lock --upgrade            # refresh the lockfile (commit the diff)
uv lock --upgrade-package X  # bump a single package
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
