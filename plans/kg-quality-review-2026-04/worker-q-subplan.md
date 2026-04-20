# Worker Q sub-plan: REC-ID-03 — Unified slug function at `lib/ontology/slugs.py`

**Branch:** `worker-q/wave4-unified-slug`
**Base:** `dev-v0.2.0` @ `3edf730`
**Effort:** M — pure refactor; tests pin behavior.
**Wave 4.1 merge gate:** Workers N, O, P are all merged (PRs #28, #29, #30).

## Goal

Consolidate three independent slug implementations into a single canonical source at
`lib/ontology/slugs.py`. Behavior-preserving: every call site produces byte-identical
output on all inputs before and after the migration.

## Byte-for-byte read of current implementations

### 1. Courseforge/scripts/generate_course.py:174–177 (canonical reference)

```python
def _slugify(text: str) -> str:
    """Convert text to a lowercase-hyphenated slug."""
    slug = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    return re.sub(r"\s+", "-", slug).strip("-")
```

Semantics:

1. Lowercase the input.
2. **Strip** (delete) any character that is not `[a-z0-9\s-]` (alnum, whitespace, hyphen).
3. Collapse runs of whitespace to a single hyphen.
4. Strip leading/trailing hyphens.

Trace `"WCAG 2.2 AA"` → lower `"wcag 2.2 aa"` → strip-non-kept removes `.` → `"wcag 22 aa"` →
ws-collapse → `"wcag-22-aa"` → strip → `"wcag-22-aa"`.

Trace `"Cognitive Load Theory"` → `"cognitive load theory"` → (unchanged) → `"cognitive-load-theory"`.

Note: digits separated by `.` fuse together (`"2.2"` → `"22"`) because `.` is removed, not
replaced. Preserved for behavior compatibility.

Empty-input trace: `""` → `""` (regex subs on empty str return empty, strip of empty stays empty).

### 2. Trainforge/process_course.py:379–392 (normalize_tag)

```python
def normalize_tag(raw: str) -> str:
    """Normalize a concept string to lowercase-hyphenated tag."""
    tag = raw.lower().strip()
    tag = re.sub(r"[^a-z0-9\s-]", "", tag)
    tag = re.sub(r"\s+", "-", tag)
    tag = tag.strip("-")
    # Limit to 4 words
    parts = tag.split("-")
    if len(parts) > 4:
        tag = "-".join(parts[:4])
    # Tags must start with a letter (LibV2 lowercase-hyphenated format)
    if tag and not tag[0].isalpha():
        return ""
    return tag
```

Semantics: the first 4 lines are byte-identical to Courseforge's `_slugify` (same lowercase + same
regex + same ws-collapse + same strip). **Verified — the master plan assumption that this was
already split into canonical + display layers by Worker H does not hold; the function still
combines canonicalization with truncation and the alpha-first rejection.**

Extra post-processing:

1. Split on `-`; truncate to at most 4 parts (display-facing cap — e.g. keeps tag URLs short).
2. Reject tags whose first character isn't alphabetic (returns `""`).

Trace `"Cognitive Load Theory"` → same base → `"cognitive-load-theory"` → 3 parts ≤ 4 → first
char `c` is alpha → `"cognitive-load-theory"`.

Trace `"2nd generation neural network model"` → lower → `"2nd generation neural network model"` →
regex pass-through → ws-collapse → `"2nd-generation-neural-network-model"` → split 5 parts →
truncate to 4 → `"2nd-generation-neural-network"` → first char `2` NOT alpha → `""`.

### 3. Trainforge/rag/inference_rules/is_a_from_key_terms.py:52–64

```python
def _slugify(text: str) -> str:
    text = canonicalize_sc_references(text or "")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text
```

Semantics:

1. Canonicalize WCAG Success Criteria references via `canonicalize_sc_references` (unique to
   this call site; e.g. normalizes "1.3.1" → "SC-1.3.1" or vice versa — see
   `Trainforge/rag/wcag_canonical_names.py`).
2. Lowercase + strip.
3. Replace non-`[a-z0-9\s\-]` characters with **SPACE** (not empty — differs from Courseforge).
4. Collapse whitespace to hyphen.
5. Collapse runs of hyphens to a single hyphen.
6. Strip edge hyphens.

Trace `"a.b"` → canonicalize → `"a.b"` (no SC ref) → lower → `"a.b"` → `.` → space → `"a b"` →
ws-collapse → `"a-b"` → multi-hyphen collapse → `"a-b"` → strip → `"a-b"`.

Contrast with Courseforge on same input: `"a.b"` → `.` deleted → `"ab"`. **Different byte output.**

This matters for the `is-a` inference rule because key-term phrases with internal dots /
parenthetical punctuation should stay word-separated after slugging to enable correct substring
matching against graph node ids.

## Canonical algorithm identification

Per the master plan line 45:

> `canonical_slug(text)` matches Courseforge's `_slugify` exactly (unlimited kebab-case).
> Trainforge's `normalize_tag` becomes a thin wrapper that calls `canonical_slug` then applies
> its display-only truncation (already split per Worker H's Wave 1 design).

Two of three call sites already share exact semantics (Courseforge `_slugify` and the base of
Trainforge `normalize_tag`). Those consolidate cleanly.

The third call site (`is_a_from_key_terms._slugify`) replaces non-alnum with SPACE instead of
deleting, then runs the same collapse pipeline. This differs semantically for inputs containing
internal punctuation. Migration strategy: preprocess punctuation → space FIRST, then call
`canonical_slug`. The preprocessed text has no punctuation (only alnum + whitespace + hyphen),
so `canonical_slug`'s delete step is a no-op and its ws-collapse does the rest — byte-identical
to the original four-line pipeline.

## Migration strategy per call site

### Call site 1: `Courseforge/scripts/generate_course.py::_slugify`

Replace the four-line function body with an import alias. Preserve the name `_slugify` so the
40-odd internal callers (lines 350, 431, 537, etc.) don't change. The alias preserves exact
semantics.

```python
from lib.ontology.slugs import canonical_slug as _slugify
```

Delete the local definition at lines 174–177.

### Call site 2: `Trainforge/process_course.py::normalize_tag`

Split into canonicalization + display-layer post-processing. Import `canonical_slug`; keep the
truncation + alpha-first rules as they were.

```python
from lib.ontology.slugs import canonical_slug

def normalize_tag(raw: str) -> str:
    """Normalize a concept string to lowercase-hyphenated tag.

    Canonicalization matches the shared `lib.ontology.slugs.canonical_slug`
    function (REC-ID-03). Display-only truncation to 4 tokens and the
    alpha-first rejection remain specific to Trainforge's LibV2 tag format.
    """
    tag = canonical_slug(raw)
    # Limit to 4 words (display cap)
    parts = tag.split("-")
    if len(parts) > 4:
        tag = "-".join(parts[:4])
    # Tags must start with a letter (LibV2 lowercase-hyphenated format)
    if tag and not tag[0].isalpha():
        return ""
    return tag
```

Byte-equivalence check: the original first four lines produce the same string as
`canonical_slug(raw)` because (a) the old code's `raw.lower().strip()` prefix is absorbed by
`canonical_slug`'s lowercase step and subsequent whitespace-collapse + edge-strip, and (b) the
regex + collapse + strip are the same operations in the same order. Verified by parametrized
regression test.

### Call site 3: `Trainforge/rag/inference_rules/is_a_from_key_terms.py::_slugify`

Keep the file-local `_slugify` name (used 3x internally at L89, L158). Refactor its body to call
`canonical_slug` after preprocessing punctuation → space.

```python
from lib.ontology.slugs import canonical_slug

def _slugify(text: str) -> str:
    """Normalize a term to a concept-graph-style id.

    Preserves this site's SC-reference canonicalization and punctuation→space
    semantics (differs from the plain canonical slug in that `"a.b"` stays
    word-separated as `"a-b"` rather than fusing to `"ab"`), then delegates
    final kebab-case normalization to the shared slug function.
    """
    text = canonicalize_sc_references(text or "")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    return canonical_slug(text)
```

Byte-equivalence trace for inputs without SC references:

- Input `"a.b"` → canonicalize (no-op) → `"a.b"` → lower + punct→space → `"a b"` →
  `canonical_slug("a b")` → (already lowercase) regex-delete (no-op on alnum+space) → ws-collapse
  → `"a-b"` → strip → `"a-b"`. Matches original `"a-b"`.
- Input `"--a--b--"` → canonicalize (no-op) → `"--a--b--"` lower → regex punct→space preserves
  hyphens → `"--a--b--"` → `canonical_slug` lowercases (no-op), regex-delete keeps the hyphens,
  ws-collapse no-op (no ws), strip → `"a--b"`. Original: `"--a--b--"` → punct→space no-op (keeps
  hyphens) → `"--a--b--"` → ws-collapse no-op → `"--a--b--"` → multi-hyphen collapse → `"-a-b-"`
  → strip → `"a-b"`.

  Divergence: new `"a--b"` vs old `"a-b"`. The old code had an explicit multi-hyphen collapse
  step; `canonical_slug` does not. Need to add that collapse to `canonical_slug` OR do it in
  the wrapper. Since Courseforge's `_slugify` ALSO lacks multi-hyphen collapse and we're
  specified to match it exactly, test `canonical_slug("--a--b--")` to see — lower → `"--a--b--"`
  → `re.sub(r"[^a-z0-9\s-]", "", ...)` keeps `-`, keeps space (none), removes nothing → still
  `"--a--b--"` → `re.sub(r"\s+", "-", ...)` no ws → unchanged → strip → `"a--b"`.

  So Courseforge `_slugify("--a--b--")` = `"a--b"`. Confirmed. `canonical_slug` must match that
  exactly (= `"a--b"`).

  The is_a call site's original explicit multi-hyphen collapse makes it produce `"a-b"` instead.
  To preserve is_a behavior we need to collapse multi-hyphens in the wrapper:

  ```python
  def _slugify(text: str) -> str:
      text = canonicalize_sc_references(text or "")
      text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
      slug = canonical_slug(text)
      return re.sub(r"-+", "-", slug).strip("-")
  ```

  Trace `"--a--b--"` with this new form: canonicalize → `"--a--b--"` → lower+punct-to-space
  keeps hyphens → `"--a--b--"` → canonical_slug → `"a--b"` → collapse `"-+"` → `"a-b"` →
  strip → `"a-b"`. Matches original.

Alternative would be to add a `collapse_multi_hyphens: bool = False` kwarg to `canonical_slug`,
but that leaks the is_a-specific detail into the canonical function. Keep the collapse in the
wrapper so `canonical_slug` stays a clean single-responsibility function matching the
Courseforge reference.

### Summary of output file after migration

- `lib/ontology/slugs.py` (new): `canonical_slug(text)` + module docstring + `__all__`.
- `lib/tests/test_slugs.py` (new): behavior-preservation regression.
- Three call-site files each gain a short import + either alias or a thin wrapper.

## Test plan

`lib/tests/test_slugs.py` covers:

1. **`test_canonical_slug_basic`** — `"Cognitive Load Theory"` → `"cognitive-load-theory"`.
2. **`test_canonical_slug_empty`** — `""` → `""`.
3. **`test_canonical_slug_none_safe`** — `None`-like / falsy input handled without crashing.
4. **`test_canonical_slug_only_punctuation`** — `"!!!"` → `""` (strip-then-collapse yields empty
   after leading/trailing hyphen strip because all `!`s are removed by the delete step; the
   result is empty before collapse, strip leaves empty).
5. **`test_canonical_slug_numbers`** — `"WCAG 2.2 AA"` → `"wcag-22-aa"` (dot stripped, digits
   fuse — pinned to Courseforge `_slugify` semantics).
6. **`test_canonical_slug_leading_trailing_hyphens`** — `"-foo-bar-"` → `"foo-bar"`.
7. **`test_canonical_slug_preserves_internal_multi_hyphens`** — `"--a--b--"` → `"a--b"` (the
   strip only removes edge hyphens; internal run preserved; matches Courseforge `_slugify`).
8. **`test_canonical_slug_case_insensitive`** — `"FooBar"` / `"FOOBAR"` both → `"foobar"`.
9. **`test_canonical_slug_whitespace_collapse`** — `"a   b"` → `"a-b"`.
10. **`test_canonical_slug_idempotent`** — for any output `s`, `canonical_slug(s) == s` on a
    range of already-slugged inputs.
11. **`test_courseforge_slugify_parity`** — parametrized sweep: import the new
    `_slugify` alias from `Courseforge.scripts.generate_course`, compare against
    `canonical_slug` for ~20 inputs, assert byte-equal.
12. **`test_trainforge_normalize_tag_parity_on_short_input`** — for inputs ≤4 tokens starting
    with a letter, `normalize_tag(x) == canonical_slug(x)`.
13. **`test_trainforge_normalize_tag_truncates_to_4_tokens`** — pin the display-layer cap.
14. **`test_trainforge_normalize_tag_rejects_numeric_first_char`** — pin the alpha-first rule.
15. **`test_is_a_slugify_parity_on_plain_input`** — for inputs with no SC references and no
    internal punctuation, `is_a._slugify(x) == canonical_slug(x)`.
16. **`test_is_a_slugify_preserves_sc_canonicalization`** — for an input containing a bare SC
    reference, the is_a slugify applies `canonicalize_sc_references` first (spot-check one
    case).
17. **`test_is_a_slugify_punctuation_stays_word_separated`** — `"a.b"` → `"a-b"` (not `"ab"`
    like Courseforge's), pinning the punctuation-to-space semantic.
18. **`test_all_three_callers_agree_on_plain_input`** — parametrized sweep of ~20 inputs
    without SC references or internal punctuation; assert `_slugify (courseforge) ==
    canonical_slug == is_a._slugify` (and `normalize_tag` matches too when input is ≤4 tokens
    starting with letter).

## Verification commands

```bash
python3 -m ci.integrity_check
source venv/bin/activate && pytest lib/tests/test_slugs.py -x -v
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q
```

## Constraints honored

- No change to `Trainforge/generators/preference_factory.py` (Worker R's scope).
- No change to `normalize_tag`'s display-truncation / alpha-first layer (only the
  canonicalization portion is factored out).
- Main branch untouched. Target `dev-v0.2.0`.
- Pure refactor — no schema changes, no new env flags.
