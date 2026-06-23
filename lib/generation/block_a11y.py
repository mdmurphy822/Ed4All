"""IB4 — ``ED4ALL_BLOCK_A11Y`` flag resolver (per-block WCAG 2.2 AA + UDL emit gate).

Single source of truth for the IB4 emit/enforcement flag. Default OFF — when
unset (or falsey / garbage), the per-block a11y sub-check in
``lib.validators.rewrite_html_shape.RewriteHtmlShapeValidator._check_block_a11y_contract``
is a no-op AND the deterministic UDL fields
(``n_representations`` / ``response_formats`` / ``engagement_affordance``) are
NOT emitted to HTML / JSON-LD, so every existing snapshot / ``contentHash`` stays
byte-identical (mirrors ``ED4ALL_BLOCK_ANATOMY`` / ``COURSEFORGE_EMIT_BLOCKS`` /
``ED4ALL_KEY_TERMS_PAGE``).

Keyboard / focus contract (the IB4.1 per-block sub-check, gated by this flag):

* **Keyboard operability (WCAG 2.1.1).** A static-HTML validator can't run JS
  ``keydown`` handlers, so the contract is structural: an interactive control
  MUST be a NATIVE interactive element (``<button>`` / ``<a href>`` /
  ``<input>`` / ``<select>`` / ``<textarea>`` / ``<details>`` / ``<summary>``)
  OR carry an explicit, documented keyboard affordance the renderer emits — a
  native key-handler attribute (``onkeydown`` / ``onkeyup`` / ``onkeypress``)
  or a ``data-cf-keys`` / ``data-cf-keyboard`` marker enumerating the keys
  (Enter / Space / Arrow) the control honors. A click-only custom control
  (e.g. a ``<div role="button" onclick=…>`` or a ``data-cf-component``-bearing
  ``<div>`` / ``<span>``) with neither is operable by mouse only and is flagged
  ``BLOCK_KEYBOARD_OPERABLE`` (warning-day-1, deferred critical-flip). A native
  ``<details>`` / ``<summary>`` reveal anywhere in the block is the escape
  hatch (keyboard-operable for free).
* **Visible focus (WCAG 2.4.7).** An interactive element (native OR custom)
  carrying an inline ``outline:none`` / ``outline:0`` in its ``style`` WITHOUT
  a replacement focus indicator (``box-shadow`` / ``border`` / ``outline-*``
  declaration in the same inline style) suppresses the keyboard-focus ring and
  is flagged ``BLOCK_FOCUS_VISIBLE`` (warning-day-1, deferred critical-flip).

Scope split (IB4.6 RECOMMENDATION, stated explicitly):

* The EMIT (UDL field stamping in ``Block.to_html_attrs`` / ``to_jsonld_entry``)
  and the IB4.1 per-block a11y sub-check are gated behind this flag.
* The ``chunk_wcag_status`` chunk-field gate (IB4.2), the ``textbook_to_course``
  packaging ``wcag_compliance`` gate (IB4.3), and the ``udl_coverage`` validator
  (IB4.5) run warning-day-1 REGARDLESS of this flag — they read existing data /
  reuse ``WCAGValidator`` and can't break a run (warning severity), so they
  provide the measurement signal the deferred critical-flips need.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else (falsey / garbage / unset) → off. Read each call so tests can
toggle the env var inline.
"""

from __future__ import annotations

import os

__all__ = ["ENV_BLOCK_A11Y", "resolve_block_a11y"]

ENV_BLOCK_A11Y = "ED4ALL_BLOCK_A11Y"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_block_a11y(value: object = None) -> bool:
    """Return True iff IB4 per-block a11y + UDL emit is enabled.

    ``value`` (optional) overrides the env var when not ``None`` — accepts the
    same truthy tokens (case-insensitive) so a caller can thread an explicit
    decision through. Falsey / garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_BLOCK_A11Y, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    try:
        token = str(raw).strip().lower()
    except Exception:  # noqa: BLE001 — never crash a resolve on a weird value
        return False
    return token in _TRUTHY
