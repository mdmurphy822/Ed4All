#!/usr/bin/env python3
"""Generate the closed Touch.provider enum into the two static sites.

The closed ``Touch.provider`` provenance set is DERIVED from the unified
endpoint registry (``config/endpoints.yaml``) via
``lib.llm.endpoints.provenance_provider_names()`` — the single authority
(union of every endpoint row's ``provenance_provider`` + the
``"deterministic"`` sentinel). The runtime site
(``Courseforge/scripts/blocks.py::_TOUCH_PROVIDERS``) imports that
function directly. The two STATIC sites can't import Python at parse
time, so this tool GENERATES them between sentinel markers:

1. ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` —
   ``$defs.Touch.properties.provider.enum`` array (PROVENANCE set).
2. ``schemas/context/courseforge_v1.shacl.ttl`` — the
   ``ed4all:provider`` ``sh:in ( ... )`` RDF list AND the ``sh:message``
   count word (PROVENANCE set, interpolated from ``len()``).
3. ``schemas/courseforge/block_routing.schema.json`` — the
   ``$defs.Spec.provider.enum`` array (ROUTING set: the routable endpoint
   NAMES + the ``openai_compatible`` alias, NOT the collapsed provenance
   union — see :func:`derived_routing_providers`).

CLI::

    python -m scripts.codegen.sync_provenance_enum            # write
    python -m scripts.codegen.sync_provenance_enum --check    # verify only

``--check`` exits non-zero (1) on any drift without writing; a bare run
rewrites the two files in place to match the derived set. The drift test
``schemas/tests/test_touch_provider_enum_sync.py`` + the
``ci/integrity_check.py`` hook both run ``--check`` so a divergence fails
the build.

Implementation discipline: the JSON edit replaces ONLY the enum array
(targeted line rewrite preserving 2-space indent + surrounding bytes);
the TTL edit replaces ONLY the ``sh:in`` list + the count word in the
following ``sh:message``. No full reformat — diffs stay minimal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# scripts/ is a PEP-420 namespace package; ensure the repo root is importable
# when invoked as `python scripts/codegen/sync_provenance_enum.py` directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.llm.endpoints import (  # noqa: E402
    endpoint_names,
    provenance_provider_names,
)

JSONLD_SCHEMA = _REPO_ROOT / "schemas" / "knowledge" / "courseforge_jsonld_v1.schema.json"
SHACL_TTL = _REPO_ROOT / "schemas" / "context" / "courseforge_v1.shacl.ttl"
BLOCK_ROUTING_SCHEMA = (
    _REPO_ROOT / "schemas" / "courseforge" / "block_routing.schema.json"
)

# Legacy provider alias the block_routing schema admits alongside the
# registry endpoint names (mirrors ``router._OPENAI_COMPATIBLE_ALIAS``).
_OPENAI_COMPATIBLE_ALIAS = "openai_compatible"


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derived_providers() -> List[str]:
    """The closed, sorted Touch.provider set from the endpoint registry.

    This is the PROVENANCE set (union of every row's ``provenance_provider``
    + ``"deterministic"``) — used by the JSON-LD + SHACL Touch sites. It
    collapses many endpoints onto one provenance value (groq/fireworks/
    deepseek → ``local``), so it is NOT the same set as
    :func:`derived_routing_providers`.
    """
    return list(provenance_provider_names())


def derived_routing_providers() -> List[str]:
    """The block_routing schema provider enum, derived from the registry.

    DISTINCT from :func:`derived_providers`: the block_routing
    ``Spec.provider`` enum is the set of routable endpoint NAMES (each
    endpoint is selectable per (block_type, tier)) plus the legacy
    ``openai_compatible`` alias — NOT the collapsed provenance set. Sorted
    for a stable, minimal diff.
    """
    return sorted(endpoint_names()) + [_OPENAI_COMPATIBLE_ALIAS]


# ---------------------------------------------------------------------------
# JSON-LD schema enum
# ---------------------------------------------------------------------------

# The Touch.provider enum is a single line; match it precisely so the
# generated array can't accidentally hit another `provider` enum elsewhere.
# Anchor on the leading indentation + the `"enum": [` of the provider field.
_JSON_ENUM_RE = re.compile(
    r'(?P<indent>[ \t]*)"enum": \[(?P<body>[^\]]*)\](?P<tail>,\n[ \t]*"description": "Provider backend)',
)


def _render_json_enum_body(providers: List[str]) -> str:
    return ", ".join(json.dumps(p) for p in providers)


def render_jsonld(text: str, providers: List[str]) -> str:
    """Return ``text`` with the Touch.provider enum array regenerated."""
    new_body = _render_json_enum_body(providers)

    def _sub(m: re.Match) -> str:
        return f'{m.group("indent")}"enum": [{new_body}]{m.group("tail")}'

    new_text, n = _JSON_ENUM_RE.subn(_sub, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"sync_provenance_enum: could not locate the Touch.provider enum "
            f"in {JSONLD_SCHEMA} (matched {n} regions, expected 1)"
        )
    return new_text


# ---------------------------------------------------------------------------
# block_routing schema Spec.provider enum
# ---------------------------------------------------------------------------

# The ``Spec.provider`` enum is a single line in block_routing.schema.json;
# anchor on the leading indent + the ``"enum": [`` immediately followed by a
# ``"description": "Backend selector`` tail so we can't hit another enum.
_ROUTING_ENUM_RE = re.compile(
    r'(?P<indent>[ \t]*)"enum": \[(?P<body>[^\]]*)\]'
    r'(?P<tail>,\n[ \t]*"description": "Backend selector)',
)


def render_block_routing(text: str, providers: List[str]) -> str:
    """Return ``text`` with the block_routing Spec.provider enum regenerated."""
    new_body = ", ".join(json.dumps(p) for p in providers)

    def _sub(m: re.Match) -> str:
        return f'{m.group("indent")}"enum": [{new_body}]{m.group("tail")}'

    new_text, n = _ROUTING_ENUM_RE.subn(_sub, text, count=1)
    if n != 1:
        raise RuntimeError(
            f"sync_provenance_enum: could not locate the Spec.provider enum "
            f"in {BLOCK_ROUTING_SCHEMA} (matched {n} regions, expected 1)"
        )
    return new_text


# ---------------------------------------------------------------------------
# SHACL sh:in list + count word
# ---------------------------------------------------------------------------

_SHACL_IN_RE = re.compile(
    r'(?P<indent>[ \t]*)sh:in \((?P<body>[^)]*)\) ;',
)
_SHACL_MSG_RE = re.compile(
    r'(?P<prefix>sh:message "Touch\.provider is required and must be one of the )'
    r'(?P<count>\d+)'
    r'(?P<suffix> canonical providers\.")',
)


def _render_shacl_in_body(providers: List[str]) -> str:
    # space-padded inside the parens, matching the existing TTL style
    return " " + " ".join(f'"{p}"' for p in providers) + " "


def render_shacl(text: str, providers: List[str]) -> str:
    """Return ``text`` with the provider sh:in list + count word regenerated.

    Scopes the ``sh:in`` rewrite to the ``ed4all:provider`` property shape
    (the only ``sh:in`` immediately preceding the provider ``sh:message``)
    so the unrelated ``ed4all:tier`` ``sh:in`` list is never touched.
    """
    # Find the provider sh:message first; rewrite the nearest preceding
    # sh:in within the same property block.
    msg_match = _SHACL_MSG_RE.search(text)
    if not msg_match:
        raise RuntimeError(
            f"sync_provenance_enum: could not locate the provider sh:message "
            f"count word in {SHACL_TTL}"
        )
    # The provider sh:in is the LAST sh:in occurring before the message.
    in_matches = list(_SHACL_IN_RE.finditer(text, 0, msg_match.start()))
    if not in_matches:
        raise RuntimeError(
            f"sync_provenance_enum: could not locate the provider sh:in list "
            f"in {SHACL_TTL}"
        )
    target = in_matches[-1]
    new_in = (
        f'{target.group("indent")}sh:in ('
        f'{_render_shacl_in_body(providers)}) ;'
    )
    text = text[: target.start()] + new_in + text[target.end():]

    # Re-find the message (offsets shifted) and rewrite the count word.
    def _msg_sub(m: re.Match) -> str:
        return f'{m.group("prefix")}{len(providers)}{m.group("suffix")}'

    text, n = _SHACL_MSG_RE.subn(_msg_sub, text, count=1)
    if n != 1:  # pragma: no cover — defensive
        raise RuntimeError("sync_provenance_enum: count-word rewrite failed")
    return text


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _plan() -> List[Tuple[Path, str, str]]:
    """Return ``[(path, current_text, generated_text), ...]`` for each site."""
    providers = derived_providers()
    out: List[Tuple[Path, str, str]] = []

    json_text = JSONLD_SCHEMA.read_text(encoding="utf-8")
    out.append((JSONLD_SCHEMA, json_text, render_jsonld(json_text, providers)))

    shacl_text = SHACL_TTL.read_text(encoding="utf-8")
    out.append((SHACL_TTL, shacl_text, render_shacl(shacl_text, providers)))

    # The block_routing schema provider enum is the ROUTABLE endpoint
    # names + the openai_compatible alias — a different (un-collapsed) set
    # from the provenance union above.
    routing_providers = derived_routing_providers()
    routing_text = BLOCK_ROUTING_SCHEMA.read_text(encoding="utf-8")
    out.append(
        (
            BLOCK_ROUTING_SCHEMA,
            routing_text,
            render_block_routing(routing_text, routing_providers),
        )
    )
    return out


def run(check: bool) -> int:
    plan = _plan()
    drift = []
    for path, current, generated in plan:
        if current != generated:
            drift.append(path)
            if not check:
                path.write_text(generated, encoding="utf-8")

    if check:
        if drift:
            for p in drift:
                print(
                    f"DRIFT: {p.relative_to(_REPO_ROOT)} does not match the "
                    f"registry-derived enum (Touch provenance "
                    f"{derived_providers()} / routing "
                    f"{derived_routing_providers()}); run "
                    f"`python -m scripts.codegen.sync_provenance_enum`",
                    file=sys.stderr,
                )
            return 1
        print(
            "OK: Touch.provider enum sites match the derived set "
            f"{derived_providers()}"
        )
        return 0

    if drift:
        for p in drift:
            print(f"WROTE: {p.relative_to(_REPO_ROOT)}")
    else:
        print("No changes — provenance enum sites already in sync.")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the static sites match the derived set; exit 1 on drift "
        "(no write).",
    )
    args = parser.parse_args(argv)
    return run(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
