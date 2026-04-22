# Worker J Sub-Plan — Wave 2 REC-TAX-01 + REC-JSL-02 (folded)

**Branch:** `worker-j/wave2-taxonomy-prereq`
**Base:** `dev-v0.2.0` @ `4159576` (after Wave 1: Workers F, G, H, I merged).
**Parallels:** Workers K (teaching-role) and L (packager default). No file overlap with those workers in Courseforge render functions, packager, or workflow config.

---

## 1. Scope recap

Fold two Wave 2 recs that both modify `Courseforge/scripts/generate_course.py::_build_page_metadata`:

- **REC-TAX-01** — Courseforge emits a course-level classification stub (`course_metadata.json`) at the generated-course output root AND a `classification` block on every page's JSON-LD. Trainforge consumes the stub when present; CLI flags (`--division`, `--domain`, `--subdomain`) become override instead of sole source. `validate_classification` runs at Courseforge emit time (fail-closed on misclassification).
- **REC-JSL-02** — Courseforge emits `prerequisitePages: [pageId, ...]` on the page JSON-LD when prerequisites are declared.

Both land through one signature change on `_build_page_metadata` plus wiring changes in `generate_week` and `generate_course`.

---

## 2. `lib/ontology/taxonomy.py` (new, mirrors `lib/ontology/bloom.py`)

### 2.1 Source of truth

`schemas/taxonomies/taxonomy.json` — committed in Wave 1 Worker S (LibV2 schema unification) and reused here. Structure (verified):

```jsonc
{
  "version": "1.0.0",
  "divisions": {
    "STEM": {
      "name": "...",
      "domains": {
        "computer-science": {
          "name": "...",
          "subdomains": {
            "software-engineering": {
              "name": "...",
              "topics": ["design-patterns", ...]
            },
            ...
          }
        },
        ...
      }
    },
    "ARTS": { ... }
  }
}
```

### 2.2 Loader API (mirrors `bloom.py`'s `lru_cache` pattern)

```python
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas" / "taxonomies" / "taxonomy.json"
)

@lru_cache(maxsize=1)
def load_taxonomy() -> Dict:
    """Load schemas/taxonomies/taxonomy.json. Cached."""

def get_valid_divisions() -> Set[str]:
    """e.g., {'STEM', 'ARTS'}."""

def get_valid_domains(division: str) -> Set[str]:
    """Valid primary_domain slugs under a division."""

def get_valid_subdomains(division: str, domain: str) -> Set[str]:
    """Valid subdomain slugs under a division/domain."""

def get_valid_topics(division: str, domain: str, subdomain: str) -> Set[str]:
    """Valid topic slugs under a division/domain/subdomain."""

def validate_classification(classification: Dict) -> List[str]:
    """Returns [] if valid, else list of human-readable error messages.

    Checks:
    - 'division' present and in get_valid_divisions()
    - 'primary_domain' present and in get_valid_domains(division)
    - 'subdomains' (optional): every entry in get_valid_subdomains(division, domain)
    - 'topics' (optional): every entry in get_valid_topics(division, domain, subdomain)
      where the topic's parent subdomain is in subdomains (or any subdomain under
      the domain if subdomains is empty)
    """
```

Loader traversal logic is lifted from `LibV2/tools/libv2/concept_vocabulary.py:112-136` (the `_load_taxonomy` method on `ConceptVocabulary`). That method walks `divisions → domains → subdomains → topics` and accumulates slugs into a flat canonical-terms set. For `lib/ontology/taxonomy.py`, we need the hierarchical structure preserved (so `get_valid_subdomains(division, domain)` works), so we re-expose the nested dicts rather than flattening. Validation is new — LibV2 doesn't currently validate against the hierarchy, it only uses the flat set for tag normalization.

**Reusable vs LibV2-specific:**
- Reusable: the traversal pattern (`divisions → domains → subdomains → topics`) and the JSON structure.
- LibV2-specific and NOT lifted: `STOPWORDS`, `INVALID_PATTERNS`, `NormalizationResult`, `analyze_corpus` — those are concept-tag vocabulary concerns, not classification-block validation concerns.

### 2.3 `lib/tests/test_taxonomy.py`

- `test_load_taxonomy` — returns a dict with `version` and `divisions` keys.
- `test_valid_divisions` — returns exactly `{"STEM", "ARTS"}`.
- `test_get_valid_domains_stem` — `"computer-science"` ∈ STEM domains.
- `test_get_valid_domains_arts` — `"design"` ∈ ARTS domains.
- `test_get_valid_domains_bad_division` — raises or returns empty.
- `test_get_valid_subdomains_cs` — `"software-engineering"` ∈ subdomains.
- `test_validate_classification_valid` — fixture `{division: "STEM", primary_domain: "computer-science", subdomains: ["software-engineering"]}` returns `[]`.
- `test_validate_classification_invalid_division` — `{"division": "BOGUS", ...}` returns non-empty error list.
- `test_validate_classification_wrong_domain` — STEM-only domain under ARTS division returns error.
- `test_validate_classification_bad_subdomain` — unknown subdomain returns error.
- `test_validate_classification_empty` — `{}` returns errors for missing required keys.
- `test_defensive_copy_semantics` — mutating returned sets doesn't pollute cache.

---

## 3. `course_metadata.json` stub shape

Emitted by `generate_course` at `output_dir` root. Shape matches the consume contract described in LibV2/CLAUDE.md §"Course Manifest" and the Trainforge course.json's `classification` block:

```json
{
  "course_code": "WCAG_201",
  "course_title": "Web Accessibility",
  "classification": {
    "division": "STEM",
    "primary_domain": "computer-science",
    "subdomains": ["software-engineering"],
    "topics": []
  },
  "ontology_mappings": {
    "acm_ccs": [],
    "lcsh": []
  }
}
```

**Field decisions:**
- Filename `course_metadata.json` (matches master plan; avoids collision with Trainforge's own `course.json` output).
- `ontology_mappings` populated but empty by default; callers may fill in later (not in Worker J scope — future rec).
- Emitted only when `classification` is non-empty at `generate_course` time (backward-compat — existing callers without classification data get no stub).

**Validation:** `validate_classification(classification)` is called BEFORE writing the stub and BEFORE writing any page. Fail-closed: raises `ValueError` with a concatenation of error messages, so no IMSCC is produced from an invalid classification.

---

## 4. JSON-LD page metadata additions

### 4.1 `classification` key on page JSON-LD (REC-TAX-01)

Emitted on every page when `classification` is provided:

```json
{
  "@context": "https://ed4all.dev/ns/courseforge/v1",
  "@type": "CourseModule",
  ...,
  "classification": {
    "division": "STEM",
    "primary_domain": "computer-science",
    "subdomains": ["software-engineering"],
    "topics": []
  }
}
```

### 4.2 `prerequisitePages` key on page JSON-LD (REC-JSL-02)

Field name `prerequisitePages` matches the schema field at `schemas/knowledge/courseforge_jsonld_v1.schema.json:58-62`:

```jsonc
"prerequisitePages": {
  "type": "array",
  "items": { "type": "string" },
  "description": "pageIds that are prerequisite to this page."
}
```

Emitted when `prerequisite_pages` arg is non-empty for a given page.

**Schema note:** The knowledge schema currently declares `additionalProperties: false` at the root. That means an emitter introducing `classification` without a schema bump would technically fail strict validation. Per the Wave 2 plan constraint ("Do NOT modify `schemas/knowledge/courseforge_jsonld_v1.schema.json`"), we proceed with the `classification` key in emit. The Wave 1 chunk_v4 validator is opt-in (`TRAINFORGE_VALIDATE_CHUNKS=true`) and the JSON-LD schema has no active hard validator yet — there's no fail-closed validator gating on this shape today. Worker C's outline (Wave 3) will revise the knowledge schema to admit `classification` at the root. Downstream consumers read JSON-LD via json parse, not schema-validated strictly; so emitting is safe and consistent with the stated field name.

---

## 5. `_build_page_metadata` diff (old → new)

**Old (L574–598):**
```python
def _build_page_metadata(
    course_code: str, week_num: int, module_type: str, page_id: str,
    objectives: Optional[List[Dict]] = None,
    sections: Optional[List[Dict]] = None,
    misconceptions: Optional[List[Dict]] = None,
    suggested_assessments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the JSON-LD metadata dict for a single page."""
    meta: Dict[str, Any] = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": course_code,
        "weekNumber": week_num,
        "moduleType": module_type,
        "pageId": page_id,
    }
    if objectives:
        meta["learningObjectives"] = _build_objectives_metadata(objectives)
    if sections:
        meta["sections"] = _build_sections_metadata(sections)
    if misconceptions:
        meta["misconceptions"] = misconceptions
    if suggested_assessments:
        meta["suggestedAssessmentTypes"] = suggested_assessments
    return meta
```

**New:**
```python
def _build_page_metadata(
    course_code: str, week_num: int, module_type: str, page_id: str,
    objectives: Optional[List[Dict]] = None,
    sections: Optional[List[Dict]] = None,
    misconceptions: Optional[List[Dict]] = None,
    suggested_assessments: Optional[List[str]] = None,
    classification: Optional[Dict] = None,             # NEW (REC-TAX-01)
    prerequisite_pages: Optional[List[str]] = None,    # NEW (REC-JSL-02)
) -> Dict[str, Any]:
    """Build the JSON-LD metadata dict for a single page.

    When ``classification`` is provided, the course's taxonomy block is
    inherited on every page (REC-TAX-01). When ``prerequisite_pages`` is
    non-empty, emits the ``prerequisitePages`` array (REC-JSL-02).
    """
    meta: Dict[str, Any] = {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": course_code,
        "weekNumber": week_num,
        "moduleType": module_type,
        "pageId": page_id,
    }
    if objectives:
        meta["learningObjectives"] = _build_objectives_metadata(objectives)
    if sections:
        meta["sections"] = _build_sections_metadata(sections)
    if misconceptions:
        meta["misconceptions"] = misconceptions
    if suggested_assessments:
        meta["suggestedAssessmentTypes"] = suggested_assessments
    if classification:
        meta["classification"] = classification
    if prerequisite_pages:
        meta["prerequisitePages"] = prerequisite_pages
    return meta
```

---

## 6. `generate_week` signature + threading

Add two optional parameters; thread them to all six `_build_page_metadata` call sites in `generate_week`:

```python
def generate_week(
    week_data: Dict,
    output_dir: Path,
    course_code: str,
    canonical_objectives: Optional[Dict[str, Any]] = None,
    classification: Optional[Dict] = None,             # NEW
    prerequisite_map: Optional[Dict[str, List[str]]] = None,  # NEW
):
```

Where `prerequisite_map: Dict[page_id, List[prereq_page_ids]]`. For each page call site, compute `prerequisite_pages = (prerequisite_map or {}).get(page_id, [])` and pass it along with `classification=classification` to `_build_page_metadata`.

Six call sites touched (overview, content modules, application, self-check, summary, discussion).

---

## 7. `generate_course` changes

### 7.1 Signature

```python
def generate_course(
    course_data_path: str,
    output_dir: str,
    objectives_path: Optional[str] = None,
    classification: Optional[Dict] = None,  # NEW: from course plan OR CLI flags
):
```

### 7.2 Flow

1. Load course data JSON.
2. If `classification` arg is None, try `data.get("classification")` from course plan. If neither, set to `None` (backward compat).
3. If `classification` is non-empty: call `validate_classification(classification)`. If errors: raise `ValueError(f"Invalid classification: {'; '.join(errors)}")` — fail-closed before any file writes.
4. Compute `prerequisite_map` from course data:
   - Read per-week `prerequisite_pages` hints if present on pages (future enhancement hook); for Wave 2 initial, the map is whatever `data.get("prerequisite_map", {})` supplies. Empty-dict default preserves backward-compat (no prerequisitePages arrays emitted).
5. For each week: `generate_week(week, out, course_code, canonical_objectives=canonical, classification=classification, prerequisite_map=prerequisite_map)`.
6. After all pages generated: if `classification` is non-empty, write `course_metadata.json` to `output_dir` root with the shape in §3.

### 7.3 CLI additions

```python
parser.add_argument("--division", default=None, choices=["STEM", "ARTS"], help="Classification division")
parser.add_argument("--primary-domain", default=None, help="Classification primary domain slug")
parser.add_argument("--subdomains", default="", help="Comma-separated subdomain slugs")
```

Build classification dict if `--division` and `--primary-domain` are both set:
```python
classification = None
if args.division and args.primary_domain:
    classification = {
        "division": args.division,
        "primary_domain": args.primary_domain,
        "subdomains": [s.strip() for s in (args.subdomains or "").split(",") if s.strip()],
        "topics": [],
    }
```

Pass to `generate_course(..., classification=classification)`. CLI flags override any `classification` declared in the course data JSON.

---

## 8. Trainforge `process_course.py` stub-consume path

### 8.1 Current behavior (L581–612, L1860–1867, L2874–2885)

- `CourseProcessor.__init__` takes `division`, `domain`, `subdomains` as positional params, defaults `"STEM"`, `""`, `[]`.
- `_generate_manifest` writes `classification = {division: self.division, primary_domain: self.domain, subdomains: self.subdomains, ...}`.
- `main()` binds CLI flags directly to constructor.

### 8.2 New behavior

Add `_load_classification_stub()` that searches for `course_metadata.json` in:
1. Inside the IMSCC zip at root (`z.read("course_metadata.json")`).
2. Alongside the IMSCC file (`self.imscc_path.parent / "course_metadata.json"`). This supports the common Courseforge layout where `generate_course.py` writes the stub to the content dir (where the IMSCC is also typically written).

Returns parsed classification dict or None.

In `__init__`, after self.division/domain/etc are initialized from kwargs, call `_load_classification_stub()` once and merge:
- If stub exists: use stub's `division`, `primary_domain`, `subdomains`, `topics` as the base.
- Then: individual kwargs (CLI-origin) override individual fields of the stub if they're non-default. (Non-default = caller explicitly passed them; default-flag detection is done at CLI parse time by comparing to parser defaults.)

Simpler approach: take an explicit `classification_override` dict at constructor level, built in `main()` from the CLI flags that were actually passed. Merge: stub as base, override keys present.

**Log emitted sources:**
- `"Using classification from course_metadata.json stub"` — stub exists, no CLI override.
- `"Using classification from CLI flags (override stub)"` — both present.
- `"Using classification from CLI flags"` — no stub, CLI only.
- `"No classification provided"` — neither — keeps backward-compat (current default behavior).

### 8.3 CLI-flag-was-passed detection

Python's argparse with `default=SENTINEL` pattern. Change the current CLI defaults from hardcoded (`"STEM"`, required domain) to None sentinels. In `main()`:
- If `args.division is None` and no stub: default to `"STEM"` (backward compat default for legacy pipelines).
- If `args.division` set: override stub's `division`.
- Same for domain, subdomain, topic.

Problem: `--domain` is currently `required=True`. Changing that breaks external callers. Keep it required for backward-compat but introduce a sentinel comparison: if `args.domain == <default>` AND stub exists, use stub's domain; else use `args.domain`. Since required args have no default, we can't easily detect "did user pass this". Safest: make `--domain` optional (`default=None`), require either CLI flag OR stub at resolution time; fail if neither.

Final rule: `--domain` optional when stub exists. Both absent → error. This is a non-breaking change for callers that passed `--domain` (they still get their value) and unlocks callers that rely on stub-only.

---

## 9. Regression tests

`Trainforge/tests/test_taxonomy_stub.py` — four tests, no real IMSCC zip construction (use minimal fixtures):

1. `test_stub_driven_classification` — create a tmp dir with a minimal IMSCC zip (imsmanifest.xml + one HTML file + course_metadata.json at root); run `CourseProcessor._load_classification_stub`; assert returns the expected classification dict.

2. `test_cli_override_stub` — stub has `division: STEM`; call with CLI `division=ARTS`; assert the resolved classification has `division: ARTS`.

3. `test_no_stub_no_cli` — minimal IMSCC without stub, no CLI flags beyond the required ones; assert backward-compat defaults kick in (division=STEM, domain=caller's value).

4. `test_stub_invalid_fails_at_emit` — call `generate_course` with a bogus classification dict (e.g., `division: BOGUS`); assert `ValueError` raised before any files written (check output_dir is empty after failure).

`lib/tests/test_taxonomy.py` — see §2.3.

---

## 10. Constraints compliance

- Touches `Courseforge/scripts/generate_course.py`: `_build_page_metadata`, `generate_week`, `generate_course`, `_build_cli_parser`. Does NOT touch `_render_flip_cards`, `_render_self_check`, `_render_activities`, `_build_sections_metadata` (Worker K's scope).
- Does NOT touch `Courseforge/scripts/package_multifile_imscc.py` (Worker L's scope).
- Does NOT touch `config/workflows.yaml` (Worker L's scope).
- Does NOT modify `schemas/taxonomies/taxonomy.json` (existing authoritative file from Wave 1).
- Does NOT modify `schemas/knowledge/courseforge_jsonld_v1.schema.json` (Wave 1 output, stable).
- Backward compat: all new params default to None/empty; existing pipelines without classification produce no stub, no classification block, no prerequisitePages.

---

## 11. Packaging note (critical caveat)

`Courseforge/scripts/package_multifile_imscc.py` currently only zips `imsmanifest.xml` and `week_*/*.html`. `course_metadata.json` lives at the content_dir root (where `generate_course.py` writes it) and will NOT be included in the IMSCC zip until a packager update lands. Since Worker J's scope forbids touching the packager, the consume path in Trainforge falls back to reading `imscc_path.parent / "course_metadata.json"` — i.e., the stub beside the IMSCC file, which is where generate_course wrote it when the content_dir and IMSCC output dir are the same.

This is the same directory layout the existing end-to-end pipeline uses (`generate_course.py` writes to an `exports/` dir, and the IMSCC is packaged to that same dir in the existing Courseforge workflow). The fallback is clean.

A future worker (not J) will add `course_metadata.json` to the IMSCC zip at packaging time. When that lands, Trainforge's zip-first lookup (§8.2 step 1) transparently picks it up.
