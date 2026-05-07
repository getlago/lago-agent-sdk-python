"""Sweep every captured Mistral model — every fixture must extract cleanly.

Mirrors test_all_models_sweep.py for Bedrock. Run shared/fixtures/capture_mistral_all.py
to refresh fixtures.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from lago_agent_sdk.adapters import extract_mistral_native

ROOT = pathlib.Path(__file__).parent / "fixtures" / "mistral_native" / "all_models"


def _all() -> list[pathlib.Path]:
    return sorted(ROOT.glob("*.json")) if ROOT.exists() else []


@pytest.mark.skipif(not _all(), reason="Mistral fixtures not captured (run shared/fixtures/capture_mistral_all.py)")
@pytest.mark.parametrize("path", _all(), ids=lambda p: p.stem)
def test_mistral_every_model(path: pathlib.Path):
    data = json.loads(path.read_text())
    model_id = data["_model_id"]
    response = data["_response"]
    u = extract_mistral_native(response, model_id=model_id)

    # Every successful chat completion has both input and output.
    assert u.input > 0, f"{model_id} returned input=0 — adapter broken or capture stale"
    assert u.output > 0, f"{model_id} returned output=0 — adapter broken or capture stale"
    # Every event we send must be tagged correctly.
    assert u.api == "native"
    assert u.provider == "mistral"
    assert u.model == model_id


@pytest.mark.skipif(not _all(), reason="fixtures missing")
def test_mistral_usage_shape_is_uniform():
    """Across every captured Mistral model, the usage shape never drifts.

    Locks in the docs/mistral-native-findings.md finding that Mistral returns
    exactly one usage shape regardless of model family or modality.
    """
    expected_top = {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details"}
    expected_inner = {"cached_tokens"}
    for path in _all():
        data = json.loads(path.read_text())
        usage = data["_response"].get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        assert set(usage.keys()) == expected_top, f"{path.stem} drifted: usage={set(usage.keys())}"
        assert set(details.keys()) == expected_inner, f"{path.stem} drifted: details={set(details.keys())}"


@pytest.mark.skipif(not _all(), reason="fixtures missing")
def test_mistral_no_image_or_audio_tokens_separately(capsys):
    """Vision-capable Mistral models do NOT break out image tokens separately —
    image tokens stay folded into prompt_tokens. Confirms our docs finding."""
    vision = [p for p in _all() if "__vision" in p.stem]
    with capsys.disabled():
        print(f"\n  vision-call fixtures: {len(vision)}  (none should expose image_tokens)")
    for path in vision:
        data = json.loads(path.read_text())
        u = extract_mistral_native(data["_response"], model_id=data["_model_id"])
        # The whole point: image_input stays 0 because the field doesn't exist.
        assert u.image_input == 0, f"{path.stem} unexpectedly populated image_input"


@pytest.mark.skipif(not _all(), reason="fixtures missing")
def test_mistral_capability_summary(capsys):
    """Print a coverage breakdown — useful as a docs anchor."""
    by_family: dict[str, int] = {}
    for path in _all():
        data = json.loads(path.read_text())
        prefix = data["_model_id"].split("-")[0]
        by_family[prefix] = by_family.get(prefix, 0) + 1
    with capsys.disabled():
        print(f"\nMistral coverage: {len(_all())} fixtures across {len(by_family)} families")
        for fam, n in sorted(by_family.items(), key=lambda x: -x[1]):
            print(f"  {fam}: {n}")
    expected = {"mistral", "open", "codestral", "devstral", "ministral", "magistral", "voxtral", "pixtral"}
    found = set(by_family.keys())
    assert expected.issubset(found), f"missing families: {expected - found}"
