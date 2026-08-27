"""LagoClient — verify_ssl passthrough.

A local dev Lago instance behind a self-signed certificate is a real, common
setup; the only alternative without this flag is routing every request
through a public tunnel purely to get a browser-trusted cert.
"""

from __future__ import annotations

from unittest.mock import patch

from lago_agent_sdk.config import LagoConfig
from lago_agent_sdk.lago_client import LagoClient
from lago_agent_sdk.sdk import LagoSDK


def test_verify_ssl_defaults_to_true() -> None:
    client = LagoClient(api_key="k", api_url="https://api.getlago.com/api/v1")
    assert client.verify_ssl is True
    # Patch the SESSION, not `requests.post` — the client keeps one Session alive so
    # the TLS handshake is not repeated per batch. Patching the module-level function
    # here silently stopped intercepting and let a real request out to the network.
    with patch.object(client._session, "post") as mock_post:
        mock_post.return_value.status_code = 200
        client.send_batch([{"transaction_id": "t1"}])
    assert mock_post.call_args.kwargs["verify"] is True


def test_verify_ssl_false_is_passed_through_to_requests() -> None:
    client = LagoClient(api_key="k", api_url="https://api.lago.dev/api/v1", verify_ssl=False)
    assert client.verify_ssl is False
    with patch.object(client._session, "post") as mock_post:
        mock_post.return_value.status_code = 200
        client.send_batch([{"transaction_id": "t1"}])
    assert mock_post.call_args.kwargs["verify"] is False


def test_lago_config_verify_ssl_defaults_to_true() -> None:
    assert LagoConfig(api_key="k").verify_ssl is True


def test_sdk_threads_verify_ssl_from_config_to_its_internal_client() -> None:
    sdk = LagoSDK(api_key="k", config=LagoConfig(api_key="k", verify_ssl=False))
    try:
        assert sdk._lago_client.verify_ssl is False
    finally:
        sdk.shutdown(timeout=1.0)

    sdk2 = LagoSDK(api_key="k")  # default config — verify_ssl stays True
    try:
        assert sdk2._lago_client.verify_ssl is True
    finally:
        sdk2.shutdown(timeout=1.0)
