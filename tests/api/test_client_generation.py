"""The generated TypeScript client must not drift from the running app.

`frontend/src/lib/api/schema.ts` is generated from `openapi.json`, which is
generated from the app. If a route or a response model changes without the
client being regenerated, the frontend compiles against a contract the server no
longer serves — and it compiles *successfully*, which is the dangerous part.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "frontend" / "src" / "lib" / "api"


def test_the_committed_schema_matches_the_application():
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import generate_api_client

    assert generate_api_client.main(["--check"]) == 0, (
        "OpenAPI drift — run: python scripts/generate_api_client.py"
    )


def test_the_generated_types_exist_and_carry_the_contract():
    schema_ts = API_DIR / "schema.ts"
    assert schema_ts.exists(), "run scripts/generate_api_client.py"
    text = schema_ts.read_text(encoding="utf-8")
    assert "SharpeObject:" in text
    # Required, not optional — `trials?:` would break the whole contract.
    assert "trials: number;" in text
    assert "trials?:" not in text


def test_the_hand_written_client_offers_no_number_formatter():
    """Formatting in the browser would be a second source of truth."""

    client_ts = (API_DIR / "client.ts").read_text(encoding="utf-8")
    assert "formatNum" in client_ts
    assert "toFixed" not in client_ts
    assert "Intl.NumberFormat" not in client_ts


def test_the_schema_json_is_valid_openapi():
    spec = json.loads((API_DIR / "openapi.json").read_text(encoding="utf-8"))
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"] == "QuantLab API"
    assert len(spec["paths"]) >= 25
