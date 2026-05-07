"""Sweep every captured Bedrock model — adapters never crash, dispatch is right."""
from __future__ import annotations

import json
import pathlib

import pytest

from lago_agent_sdk.adapters import (
    extract_bedrock_converse,
    extract_bedrock_invoke,
    pick_invoke_adapter,
)

ROOT = pathlib.Path(__file__).parent / "fixtures" / "bedrock"
CONV = ROOT / "converse"
INV = ROOT / "invoke"


def _all(p: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p.glob("*.json")) if p.exists() else []


@pytest.mark.skipif(not _all(CONV), reason="Converse fixtures not captured (run shared/fixtures/capture.py)")
@pytest.mark.parametrize("path", _all(CONV), ids=lambda p: p.stem)
def test_converse_every_model(path: pathlib.Path):
    data = json.loads(path.read_text())
    model_id, response = data["_model_id"], data["_response"]
    u = extract_bedrock_converse(response, model_id=model_id)
    # Every Converse response must yield non-zero input + output.
    assert u.input > 0, f"{model_id} returned input=0 — adapter broken or capture stale"
    assert u.output > 0, f"{model_id} returned output=0 — adapter broken or capture stale"
    assert u.api == "bedrock_converse"


@pytest.mark.skipif(not _all(INV), reason="InvokeModel fixtures not captured (run shared/fixtures/capture.py)")
@pytest.mark.parametrize("path", _all(INV), ids=lambda p: p.stem)
def test_invoke_every_model(path: pathlib.Path):
    data = json.loads(path.read_text())
    model_id, response = data["_model_id"], data["_response"]
    family = pick_invoke_adapter(model_id)
    u = extract_bedrock_invoke(response, model_id=model_id)
    assert u.api == "bedrock_invoke"

    if family == "mistral_legacy":
        # Spec — these models cannot be billed via InvokeModel
        assert u.extras.get("_no_usage") is True or not u.nonzero_numeric()
    else:
        # Everything else MUST yield input + output.
        assert u.input > 0, f"{model_id} ({family}) returned input=0"
        assert u.output > 0, f"{model_id} ({family}) returned output=0"


@pytest.mark.skipif(not _all(INV), reason="InvokeModel fixtures not captured")
def test_invoke_dispatch_distribution_summary(capsys):
    """Print a per-family count — sanity check coverage across families."""
    counts: dict[str, int] = {}
    for p in _all(INV):
        data = json.loads(p.read_text())
        family = pick_invoke_adapter(data["_model_id"])
        counts[family] = counts.get(family, 0) + 1
    with capsys.disabled():
        print("\nInvokeModel families covered by fixtures:")
        for f, n in sorted(counts.items()):
            print(f"  {f}: {n}")
    # Must cover at least the 5 non-empty families that actually return usage
    expected = {"openai_compat_basic", "openai_compat_with_details", "anthropic", "opus_4_7", "nova"}
    assert expected.issubset(set(counts.keys())), f"missing families: {expected - set(counts.keys())}"
