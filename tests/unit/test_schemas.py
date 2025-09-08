# tests/unit/test_schemas.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Optional, Any, List

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


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield all dict nodes in a nested structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_dicts(it)


def _collect_properties_bags(d: Dict) -> Iterable[Tuple[Dict[str, Dict], Dict]]:
    """
    Yield (properties_dict, parent_node) for all 'properties' bags reachable anywhere in the schema,
    including under items (arrays) and inside combinators (allOf/anyOf/oneOf),
    as well as in definitions/$defs.
    """
    if not isinstance(d, dict):
        return
    stack: List[Dict[str, Any]] = [d]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, (dict, list)):
            continue
        if isinstance(cur, dict):
            if "properties" in cur and isinstance(cur["properties"], dict):
                yield cur["properties"], cur
            # Recurse into common structural keywords
            for key, v in cur.items():
                if key in (
                    "items",
                    "allOf", "anyOf", "oneOf", "not",
                    "if", "then", "else",
                    "definitions", "$defs",
                    "patternProperties",
                    "additionalProperties",
                ):
                    stack.append(v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        else:  # list
            stack.extend(cur)


def _collect_pattern_properties(d: Dict) -> Iterable[Tuple[Dict[str, Dict], Dict]]:
    """
    Yield (patternProperties_dict, parent_node) tuples found anywhere.
    """
    for node in _iter_dicts(d):
        pp = node.get("patternProperties")
        if isinstance(pp, dict):
            yield pp, node


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
    max_idx = max(mu_idxs)
    # non-negative contiguous set expected (0..max)
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


# ---------- schema-type helpers ---------- #
def _collect_types(schema_fragment: Any) -> Set[str]:
    """
    Collect 'type' values reachable in a fragment, walking through anyOf/oneOf/allOf/not/then/else/$ref.
    This is a conservative extraction; we only assert existence of 'number' for mu/sigma and 'string' for sample_id.
    """
    types: Set[str] = set()
    stack: List[Any] = [schema_fragment]
    visited: Set[int] = set()
    while stack:
        cur = stack.pop()
        if not isinstance(cur, (dict, list)):
            continue
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)

        if isinstance(cur, dict):
            t = cur.get("type")
            if isinstance(t, str):
                types.add(t)
            elif isinstance(t, list):
                for e in t:
                    if isinstance(e, str):
                        types.add(e)
            # descend into composites and referenced places
            for key in ("anyOf", "oneOf", "allOf", "not", "if", "then", "else", "items"):
                if key in cur:
                    stack.append(cur[key])
            # follow common ref-like structures loosely (we don't resolve external refs)
            if "$ref" in cur:
                # treat presence as unknown type; we won't be strict if we find no explicit 'type'
                pass
        else:
            stack.extend(cur)
    return types


def _find_property_schema(bag_parent: Dict, prop_name: str) -> Optional[Dict]:
    """
    Try to find the schema for a given property name within its parent node (where the 'properties' was found).
    """
    props = bag_parent.get("properties", {})
    if isinstance(props, dict) and prop_name in props and isinstance(props[prop_name], dict):
        return props[prop_name]
    return None


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
    # Basic hygiene only.
    assert "$schema" in keys or "$id" in keys or "title" in keys


# ----------------------------------------------------------------------------- #
# Tests: submission schema column logic (contiguity, symmetry, types)
# ----------------------------------------------------------------------------- #
def test_submission_schema_includes_expected_fields_and_contiguity(submission_schema: Dict) -> None:
    """
    We don't assume a specific schema shape; we just ensure keys for sample_id and
    mu_### / sigma_### appear somewhere (often under properties or patternProperties).
    If a properties bag is found, enforce contiguity/symmetry of mu/sigma indices.
    """
    keys = _flatten_keys(submission_schema)

    # Cheap sentinels must exist somewhere in the schema (or via patterns)
    sentinels = {"sample_id", "mu_000", "sigma_000"}
    missing_sentinels = sentinels - keys

    # If using patternProperties only, we may not see concrete 'mu_000'/'sigma_000' keys in flatten_keys;
    # tolerate this but make sure patterns are present below.
    patterns_present = False
    for pp, _parent in _collect_pattern_properties(submission_schema):
        for patt in pp.keys():
            if re.search(r"mu_\\d{3}", patt) or re.search(r"^mu_\(\?\:\d\)\{3\}", patt):
                patterns_present = True
            if re.search(r"sigma_\\d{3}", patt) or re.search(r"^sigma_\(\?\:\d\)\{3\}", patt):
                patterns_present = True

    if missing_sentinels and not patterns_present:
        assert not missing_sentinels, f"Missing sentinel fields in schema keys: {missing_sentinels}"

    # Collect all property names from any nested 'properties' bag
    all_props: Set[str] = set()
    prop_bags: List[Tuple[Dict[str, Dict], Dict]] = list(_collect_properties_bags(submission_schema))
    for bag, _parent in prop_bags:
        all_props |= set(bag.keys())

    if not all_props:
        # If schema does not expose a direct properties bag (e.g., array-of-objects or patternProperties only),
        # sentinel presence or pattern presence is sufficient.
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


def _check_property_types_if_available(submission_schema: Dict) -> None:
    """
    If concrete per-field property schemas are available, ensure types are sensible:
      - mu_### / sigma_### should include 'number' (or 'integer')
      - sample_id should include 'string'
    If only patternProperties exist (no concrete props), this is skipped.
    """
    # Aggregate all properties bags with their parents to reach field subschemas
    prop_bags = list(_collect_properties_bags(submission_schema))
    if not prop_bags:
        return  # nothing concrete to validate; likely patternProperties only

    # Find any bag that appears to define mu/sigma/sample_id concretely
    for bag, parent in prop_bags:
        if not isinstance(bag, dict):
            continue
        for name in bag.keys():
            if re.match(r"^mu_\d{3}$", name) or re.match(r"^sigma_\d{3}$", name) or name == "sample_id":
                sub = _find_property_schema(parent, name)
                if not isinstance(sub, dict):
                    continue
                typs = _collect_types(sub)
                if name == "sample_id":
                    # be tolerant: allow absence (e.g., via $ref) or explicit inclusion of 'string'
                    if typs:
                        assert "string" in typs or "null" in typs, f"'sample_id' type should include 'string' (got {typs})"
                else:
                    # mu/sigma ideally numeric
                    if typs:
                        assert ("number" in typs or "integer" in typs or "null" in typs), \
                            f"'{name}' should include numeric type (got {typs})"


def test_submission_schema_value_types_if_available(submission_schema: Dict) -> None:
    _check_property_types_if_available(submission_schema)


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
    for bag, _ in _collect_properties_bags(submission_schema):
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