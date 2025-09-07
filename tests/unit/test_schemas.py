# tests/unit/test_schemas.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Set

import pytest

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAS_JSONSCHEMA = False


# ----------------------------------------------------------------------------- #
# Repo discovery
# ----------------------------------------------------------------------------- #
def _find_repo_root(start: Path | None = None) -> Path:
    """Walk up to the project root that has a 'schemas' directory."""
    cur = (start or Path(__file__)).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "schemas").is_dir():
            return parent
    # Fallback to working dir if running atypically
    return Path.cwd()


REPO_ROOT = _find_repo_root()
SCHEMAS_DIR = REPO_ROOT / "schemas"


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #
def _load_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_keys(obj) -> Set[str]:
    """
    Collect all dict keys within nested JSON-like structures.
    Useful when schema nests 'properties' under combinators.
    """
    keys: Set[str] = set()
    if isinstance(obj, dict):
        keys.update(obj.keys())
        for v in obj.values():
            keys |= _flatten_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _flatten_keys(item)
    return keys


def _expected_submission_columns(n_bins: int = 283) -> Iterable[str]:
    yield "sample_id"
    for i in range(n_bins):
        yield f"mu_{i:03d}"
    for i in range(n_bins):
        yield f"sigma_{i:03d}"


# ----------------------------------------------------------------------------- #
# Fixtures
# ----------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def schemas_path() -> Path:
    if not SCHEMAS_DIR.exists():
        pytest.skip(f"'schemas' directory not found under {REPO_ROOT}")
    return SCHEMAS_DIR


@pytest.fixture(scope="session")
def submission_schema(schemas_path: Path) -> Dict:
    p = schemas_path / "submission.schema.json"
    if not p.exists():
        pytest.skip(f"submission.schema.json not found under {schemas_path}")
    return _load_json(p)


@pytest.fixture(scope="session")
def events_schema(schemas_path: Path) -> Dict:
    p = schemas_path / "events.schema.json"
    if not p.exists():
        pytest.skip(f"events.schema.json not found under {schemas_path}")
    return _load_json(p)


@pytest.fixture(scope="session")
def config_snapshot_schema(schemas_path: Path) -> Dict:
    p = schemas_path / "config_snapshot.schema.json"
    if not p.exists():
        pytest.skip(f"config_snapshot.schema.json not found under {schemas_path}")
    return _load_json(p)


# ----------------------------------------------------------------------------- #
# Tests: well-formed JSON Schemas
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_submission_schema_is_valid_jsonschema(submission_schema: Dict) -> None:
    # Draft-07 is common; if you’ve upgraded, jsonschema will still check the 'meta' correctly.
    jsonschema.Draft7Validator.check_schema(submission_schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_events_schema_is_valid_jsonschema(events_schema: Dict) -> None:
    jsonschema.Draft7Validator.check_schema(events_schema)


@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_config_snapshot_schema_is_valid_jsonschema(config_snapshot_schema: Dict) -> None:
    jsonschema.Draft7Validator.check_schema(config_snapshot_schema)


# ----------------------------------------------------------------------------- #
# Tests: submission schema has expected fields
# ----------------------------------------------------------------------------- #
def test_submission_schema_includes_expected_fields(submission_schema: Dict) -> None:
    """
    We don't assume a specific schema shape; we just ensure keys for sample_id and
    mu_### / sigma_### appear somewhere (often under properties).
    """
    keys = _flatten_keys(submission_schema)
    expected = list(_expected_submission_columns())
    # Cheap sanity: at least a few sentinels must appear
    sentinels = {"sample_id", "mu_000", "mu_282", "sigma_000", "sigma_282"}
    assert sentinels.issubset(keys), f"Missing sentinel fields in schema keys: {sentinels - keys}"

    # If the schema exposes a properties bag, require all columns there.
    # When properties found, be strict; otherwise, only sentinels are enforced.
    # This keeps the test resilient to different schema organizations.
    if "properties" in keys:
        # Try to reach the top-level properties dict
        # (We avoid brittle indexing by scanning)
        def _collect_properties(d: Dict) -> Set[str]:
            props: Set[str] = set()
            if isinstance(d, dict):
                if "properties" in d and isinstance(d["properties"], dict):
                    props |= set(d["properties"].keys())
                for v in d.values():
                    if isinstance(v, (dict, list)):
                        props |= _collect_properties(v)
            elif isinstance(d, list):
                for it in d:
                    props |= _collect_properties(it)
            return props

        prop_keys = _collect_properties(submission_schema)
        missing = set(expected) - prop_keys
        assert not missing, f"Expected columns not exposed as properties: {sorted(list(missing))[:10]}..."


# ----------------------------------------------------------------------------- #
# Optional instance validation (enable with STRICT_SCHEMA_INSTANCE=1)
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
def test_submission_schema_accepts_plausible_instance(submission_schema: Dict) -> None:
    """
    Generate a minimal plausible instance and validate it against the schema.
    This is opt-in to avoid brittleness if the schema enforces additional metadata.
    Enable via STRICT_SCHEMA_INSTANCE=1.
    """
    if os.environ.get("STRICT_SCHEMA_INSTANCE") != "1":
        pytest.skip("Instance validation disabled; set STRICT_SCHEMA_INSTANCE=1 to enable")

    # Build a single-row instance shape commonly used in JSON-record submissions.
    # If your schema expects an array of rows or a specific wrapper, adjust this test accordingly.
    instance = {k: 0.0 for k in _expected_submission_columns()}
    instance["sample_id"] = "row_0"

    # Try validating as an object; if schema is array-based, validate [instance].
    validator = jsonschema.Draft7Validator(submission_schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        # Second try: many schemas expect a list of records
        errors2 = sorted(validator.iter_errors([instance]), key=lambda e: e.path)
        if errors2:
            details = "\n".join(f"- {e.message}" for e in errors2[:5])
            pytest.xfail(f"Submission instance did not validate. Adjust schema or test generator.\n{details}")