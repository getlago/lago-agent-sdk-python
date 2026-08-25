"""Committed fixtures must carry no personal or real-account data.

These fixtures are captured from live provider and gateway calls, and both repos
publish to a PUBLIC package index. A capture therefore arrives carrying whatever the
provider chose to log about the operator who made it — `system.ai_gateway.usage`
records the caller's account email and source IP on every row, and Snowflake's Cortex
views name the caller in ROLE_NAMES and carry a customer-controlled QUERY_TAG — none of
which any adapter reads. Twenty-two Databricks fixtures shipped with a personal Gmail address, a
residential IP, a real workspace subdomain and one live Lago subscription id before
this test existed.

There is no capture script for the gateway fixtures (they come out of a SQL warehouse
query), so there is nowhere to put a scrub step that a future recapture would run.
This test is the guard instead: it fails on the way back in.

Kept in step with `fixture_hygiene.test.ts` in the JS port.
"""

from __future__ import annotations

import ipaddress
import pathlib
import re

FIXTURE_ROOT = pathlib.Path(__file__).parent

# Only `example.com` / `example.org` — the RFC 2606 reserved names — are acceptable.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# A Databricks workspace subdomain is a real, addressable host.
_DBX_HOST = re.compile(r"\bdbc-[0-9a-f]{4,}-[0-9a-f]{4,}\b")

# Snowflake's ACCOUNT_USAGE views name the caller in ROLE_NAMES as `USER$<login>`, and a
# Snowflake account hostname is a real, addressable host. No adapter reads either.
_SF_USER_ROLE = re.compile(r"\bUSER\$[A-Za-z0-9_]+")
_SF_HOST = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*\.snowflakecomputing\.com\b")

_ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net"}
_PLACEHOLDER_DBX_HOST = "dbc-00000000-0000"
_PLACEHOLDER_SF_USER = "USER$EXAMPLE_USER"
_PLACEHOLDER_SF_HOST = "example-account.snowflakecomputing.com"


def _fixtures() -> list[pathlib.Path]:
    return sorted(FIXTURE_ROOT.rglob("*.json"))


def _is_public_ip(text: str) -> bool:
    """True only for a globally routable address — a real host somewhere."""
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    # RFC 5737 documentation ranges are the intended replacements and are not global.
    return addr.is_global


def test_no_real_email_addresses() -> None:
    offenders = [
        f"{p.relative_to(FIXTURE_ROOT)}: {domain}"
        for p in _fixtures()
        for domain in _EMAIL.findall(p.read_text(encoding="utf-8"))
        if domain.lower() not in _ALLOWED_EMAIL_DOMAINS
    ]
    assert not offenders, (
        "fixtures carry email addresses outside the RFC 2606 reserved domains — "
        f"replace with an example.com address: {offenders}"
    )


def test_no_publicly_routable_ip_addresses() -> None:
    offenders = [
        f"{p.relative_to(FIXTURE_ROOT)}: {ip}"
        for p in _fixtures()
        for ip in set(_IPV4.findall(p.read_text(encoding="utf-8")))
        if _is_public_ip(ip)
    ]
    assert not offenders, (
        "fixtures carry globally routable IP addresses — replace with an RFC 5737 "
        f"documentation address such as 203.0.113.10: {offenders}"
    )


def test_no_real_databricks_workspace_hosts() -> None:
    offenders = [
        f"{p.relative_to(FIXTURE_ROOT)}: {host}"
        for p in _fixtures()
        for host in set(_DBX_HOST.findall(p.read_text(encoding="utf-8")))
        if host != _PLACEHOLDER_DBX_HOST
    ]
    assert not offenders, (
        f"fixtures name a real Databricks workspace — use {_PLACEHOLDER_DBX_HOST}: {offenders}"
    )


def test_no_real_snowflake_users() -> None:
    offenders = [
        f"{p.relative_to(FIXTURE_ROOT)}: {name}"
        for p in _fixtures()
        for name in set(_SF_USER_ROLE.findall(p.read_text(encoding="utf-8")))
        if name != _PLACEHOLDER_SF_USER
    ]
    assert not offenders, f"fixtures name a real Snowflake user — use {_PLACEHOLDER_SF_USER}: {offenders}"


def test_no_real_snowflake_account_hosts() -> None:
    offenders = [
        f"{p.relative_to(FIXTURE_ROOT)}: {host}"
        for p in _fixtures()
        for host in set(_SF_HOST.findall(p.read_text(encoding="utf-8")))
        if host != _PLACEHOLDER_SF_HOST
    ]
    assert not offenders, f"fixtures name a real Snowflake account — use {_PLACEHOLDER_SF_HOST}: {offenders}"
