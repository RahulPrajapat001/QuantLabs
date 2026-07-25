"""Generate the frontend's typed API client from the OpenAPI schema.

The schema is the contract, and the contract is defined by the Pydantic response
models — so the TypeScript types are *generated*, never hand-written. A
hand-maintained interface drifts the moment somebody adds a field, and the drift
is invisible until a runtime `undefined`.

What this produces in ``frontend/src/lib/api/``:

* ``openapi.json`` — the schema as the running app reports it
* ``schema.ts``    — generated types (via ``openapi-typescript``)

``client.ts`` is hand-written and committed: it is a thin ``fetch`` wrapper whose
signatures reference the generated types, so it is small enough to read and it
breaks loudly at compile time when the schema changes underneath it.

Run with ``--check`` to verify the committed schema matches the app — this is
what CI does, so a route added without regenerating the client fails the build.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "frontend" / "src" / "lib" / "api"
SCHEMA_JSON = API_DIR / "openapi.json"
SCHEMA_TS = API_DIR / "schema.ts"


def openapi_schema() -> dict[str, object]:
    """The schema exactly as the running application reports it."""

    from quantlab.api.app import create_app

    return create_app().openapi()


def _serialise(schema: dict[str, object]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def generate_types() -> bool:
    """Run ``openapi-typescript``. Returns False when npx is unavailable."""

    # shutil.which resolves npx.cmd on Windows; subprocess will not find a bare
    # "npx" there, because CreateProcess does not consult PATHEXT.
    npx = shutil.which("npx")
    if npx is None:
        print(
            "npx not found — skipping schema.ts generation. openapi.json is "
            "written, so a machine with node can regenerate the types.",
            file=sys.stderr,
        )
        return False
    subprocess.run(
        [
            npx,
            "--yes",
            "openapi-typescript@7",
            str(SCHEMA_JSON),
            "-o",
            str(SCHEMA_TS),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed schema matches the app. Exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    schema = openapi_schema()
    serialised = _serialise(schema)

    if args.check:
        if not SCHEMA_JSON.exists():
            print(f"{SCHEMA_JSON} is missing — run this script.", file=sys.stderr)
            return 1
        if SCHEMA_JSON.read_text(encoding="utf-8") != serialised:
            print(
                "OPENAPI DRIFT — frontend/src/lib/api/openapi.json no longer "
                "matches the application. A route or response model changed "
                "without regenerating the client:\n"
                "  python scripts/generate_api_client.py",
                file=sys.stderr,
            )
            return 1
        print("OpenAPI schema is in sync.")
        return 0

    API_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_JSON.write_text(serialised, encoding="utf-8")
    print(f"wrote {SCHEMA_JSON.relative_to(REPO_ROOT)}")
    operations = sum(len(v) for v in schema["paths"].values())  # type: ignore[union-attr]
    print(f"  {len(schema['paths'])} paths, {operations} operations")  # type: ignore[arg-type]
    if generate_types():
        print(f"wrote {SCHEMA_TS.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
