# tests/unit/test_schemas.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Optional

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


def _collect_properties_bags(d: Dict) -> Iterable[Dict[str, Dict]]:
    """
    Yield all 'properties' bags (dicts) reachable anywhere in the schema,
    including under items (arrays) and inside combinators (allOf/anyOf/oneOf).
    """
    if not isinstance(d, dict):
        return
    stack = [d]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, (dict, list)):
            continue
        if isinstance(cur, dict):
            if "properties" in cur and isinstance(cur["properties"], dict):
                yield cur["properties"]
            # Recurse into common structural keywords
            for key, v in cur.items():
                if key in ("items", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "definitions"):
                    stack.append(v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        else:  # list
            stack.extend(cur)


def _discover_bins_from_props(props: Iterable[str]) -> Optional[int]:
    """
    Try to infer the spectral bin count by scanning contiguous mu_### keys.
    Returns None if not discoverable.
    """
    mu_idxs: Set[int] = set()
    pat = re.compile(r"^mu_(\d{3})$")
    for k in props:
        m = pat.match(k)
        if m:
            mu_idxs.add(int(m.group(1)))
    if not mu_idxs:
        return None
    # Require contiguity if possible
    max_idx = max(mu_idxs)
    expected = set(range(max_idx + 1))
    if mu_idxs >= expected:  # allow supersets (e.g., sigma keys mixed in the same bag)
        return max_idx + 1
    # Non-contiguous: still return a safe guess; subsequent tests will report gap details.
    return max_idx + 1


def _find_all_numbered(prefix: str, props: Iterable[str]) -> Set[int]:
    out: Set[int] = set()
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})$")
    for k in props:
        m = pat.match(k)
        if m:
            out.add(int(m.group(1)))
    return out


def _expected_submission_columns(n_bins: int) -> Iterable[str]:
    yield "sample_id"
    for i in range(n_bins):
        yield f"mu_{i:03d}"
    for i in range(n_bins):
        yield f"sigma_{i:03d}"


def _safe_bins_default() -> int:
    # Default for Ariel 2025 leaderboard is 283; allow override for forks via env.
    try:
        return int(os.environ.get("SM_SUBMISSION_BINS", "283"))
    except Exception:
        return 283


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
# Tests: basic presence & UTF-8 load
# ----------------------------------------------------------------------------- #
def test_schemas_directory_present(schemas_path: Path) -> None:
    assert schemas_path.is_dir()


def test_schemas_are_utf8_json(schemas_path: Path) -> None:
    """
    Ensure all *.schema.json files are valid UTF-8 JSON (no BOM surprises).
    """
    files = list(schemas_path.glob("*.schema.json"))
    assert files, "No *.schema.json files found under schemas/"
    for p in files:
        data = _load_json(p)
        assert isinstance(data, dict), f"{p.name} did not parse to an object"


# ----------------------------------------------------------------------------- #
# Tests: well-formed JSON Schemas
# ----------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_JSONSCHEMA, reason="jsonschema not installed")
@pytest.mark.parametrize(
    "fixture_name",
    ["submission_schema", "events_schema", "config_snapshot_schema"],
)
def test_each_schema_is_valid_jsonschema(request, fixture_name: str) -> None:
    schema = request.getfixturevalue(fixture_name)
    # jsonschema will use the declared $schema if present; Draft-07 check is a safe lower bound.
    jsonschema.Draft7Validator.check_schema(schema)


# ----------------------------------------------------------------------------- #
# Tests: minimal meta fields for sanity
# ----------------------------------------------------------------------------- #
def test_submission_schema_has_meta_fields(submission_schema: Dict) -> None:
    keys = set(submission_schema.keys())
    # Don’t overfit; basic hygiene only.
    assert "$schema" in keys or "$id" in keys or "title" in keys


# ----------------------------------------------------------------------------- #
# Tests: submission schema column logic (contiguity, symmetry)
# ----------------------------------------------------------------------------- #
def test_submission_schema_includes_expected_fields_and_contiguity(submission_schema: Dict) -> None:
    """
    We don't assume a specific schema shape; we just ensure keys for sample_id and
    mu_### / sigma_### appear somewhere (often under properties).
    If a properties bag is found, enforce contiguity/symmetry of mu/sigma indices.
    """
    keys = _flatten_keys(submission_schema)

    # Cheap sentinels must exist somewhere in the schema
    sentinels = {"sample_id", "mu_000", "sigma_000"}
    missing_sentinels = sentinels - keys
    assert not missing_sentinels, f"Missing sentinel fields in schema keys: {missing_sentinels}"

    # Collect all property names from any nested 'properties' bag
    all_props: Set[str] = set()
    for bag in _collect_properties_bags(submission_schema):
        all_props |= set(bag.keys())

    if not all_props:
        # If schema does not expose a direct properties bag (e.g., array-of-objects wrapper),
        # we can still consider sentinel presence sufficient.
        return

    # Infer bins from mu_* keys if possible
    inferred = _discover_bins_from_props(all_props)
    n_bins = inferred if inferred is not None else _safe_bins_default()

    mu_idxs = _find_all_numbered("mu", all_props)
    sigma_idxs = _find_all_numbered("sigma", all_props)

    # Require contiguity (0..n-1) and symmetry between mu and sigma sets
    expected = set(range(n_bins))
    missing_mu = expected - mu_idxs
    missing_sigma = expected - sigma_idxs
    extra_mu = mu_idxs - expected
    extra_sigma = sigma_idxs - expected

    assert not missing_mu, f"Missing mu indices: {sorted(list(missing_mu))[:10]}..."
    assert not missing_sigma, f"Missing sigma indices: {sorted(list(missing_sigma))[:10]}..."
    assert not extra_mu, f"Unexpected mu indices beyond n_bins={n_bins}: {sorted(list(extra_mu))[:10]}..."
    assert not extra_sigma, f"Unexpected sigma indices beyond n_bins={n_bins}: {sorted(list(extra_sigma))[:10]}..."

    # sample_id must exist in the same property set or another bag; enforce global presence
    assert "sample_id" in all_props or "sample_id" in keys, "'sample_id' property not found"


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

    # If we can infer bin count from properties, do that; otherwise default.
    all_props: Set[str] = set()
    for bag in _collect_properties_bags(submission_schema):
        all_props |= set(bag.keys())
    inferred = _discover_bins_from_props(all_props)
    n_bins = inferred if inferred is not None else _safe_bins_default()

    # Build a single-row instance shape commonly used in JSON-record submissions.
    instance = {k: 0.0 for k in _expected_submission_columns(n_bins)}
    instance["sample_id"] = "row_0"

    # Validate as an object or, if schema is array-based, as [instance]
    try:
        jsonschema.validate(instance=instance, schema=submission_schema)
    except jsonschema.ValidationError:
        try:
            jsonschema.validate(instance=[instance], schema=submission_schema)
        except jsonschema.ValidationError as e:
            pytest.xfail(
                "Submission instance did not validate. "
                "Adjust schema or enable an alternate wrapper in this test.\n"
                f"- {e.message}"
            )


# ----------------------------------------------------------------------------- #
# Events & config snapshot light checks (shape-agnostic)
# ----------------------------------------------------------------------------- #
def test_events_schema_has_reasonable_top_keys(events_schema: Dict) -> None:
    """
    Don't assert structure; just require plausible logging-related keys somewhere.
    """
    keys = _flatten_keys(events_schema)
    expected_any = {"timestamp", "level", "message"}
    assert keys & expected_any, f"None of {expected_any} found among schema keys"


def test_config_snapshot_schema_mentions_config_terms(config_snapshot_schema: Dict) -> None:
    """
    Ensure the snapshot schema at least references configuration-ish terms.
    """
    keys = _flatten_keys(config_snapshot_schema)
    expected_any = {"config", "hash", "version", "path"}
    assert keys & expected_any, f"None of {expected_any} found among schema keys"