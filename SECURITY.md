# Security

## Reporting a vulnerability

**Please don't open a public GitHub issue.** Email `security@getlago.com` instead.

We aim to respond within 2 business days and to ship a fix or mitigation within 30 days for any confirmed issue. Coordinated disclosure is appreciated.

If you'd like to send an encrypted report, ask for our PGP key in your initial mail.

## Scope

- Anything in `src/lago_agent_sdk/`
- HTTP request construction in `lago_client.py` (event payload signing, auth header handling, etc.)
- Error policy gaps (e.g. instrumentation that breaks the customer's call)

## Out of scope

- Issues in `boto3`, `mistralai`, `requests`, or other dependencies — please report those upstream
- Lago's own API security — that goes through `security@getlago.com` for the platform, not the SDK
- Customer-side misuse (e.g., logging API keys via `on_error`)

## Versions covered

We patch the latest minor of each supported major. Pre-0.1 development versions are not security-supported — pin to a release.
