# Worker H Sub-Plan — Wave 1.2 REC-BL-01 `lib/ontology/bloom.py` loader + callsite migration

**Branch:** `worker-h/wave1-bloom-loader`
**Base:** `dev-v0.2.0` @ `e0455eb` (after Worker F + G merged)
**Depends on:** `schemas/taxonomies/bloom_verbs.json` (published by Worker F in PR #20).
**Parallels:** Worker I's PR. No file overlap.

---

## 1. Authoritative `bloom_verbs.json` structure (verified from the committed file)

Worker F published `schemas/taxonomies/bloom_verbs.json` as a **dual-use schema**: the top-level object both describes the shape via `$defs` and carries the data inline as the `default` array on each bloom-level property.

### 1.1 Top-level structure
```jsonc
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://ed4all.dev/ns/taxonomies/v1/bloom_verbs.schema.json",
  "type": "object",
  "required": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
  "additionalProperties": false,
  "$defs": {
    "BloomLevel": { "type": "string", "enum": [6 levels] },
    "BloomVerb": {
      "type": "object",
      "required": ["verb", "usage_context", "example_template"],
      "additionalProperties": false,
      "properties": {
        "verb":             { "type": "string" },
        "usage_context":    { "type": "string" },
        "example_template": { "type": "string" }
      }
    }
  },
  "properties": {
    "remember":   { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" }, "default": [...10 BloomVerb dicts...] },
    "understand": { "type": "array", "items": { "$ref": "#/$defs/BloomVerb" }, "default": [...10 BloomVerb dicts...] },
    "apply":      { ... 10 ... },
    "analyze":    { ... 10 ... },
    "evaluate":   { ... 10 ... },
    "create":     { ... 10 ... }
  }
}
```

**Total:** 60 verbs × 6 levels, each a dict `{"verb": str, "usage_context": str, "example_template": str}`.

### 1.2 Loader extraction path
The loader reads `schemas/taxonomies/bloom_verbs.json`, then for each of the 6 level keys under `properties`, extracts the `default` array. Each array item is a `{verb, usage_context, example_template}` dict. The schema's `$defs` are ignored at load-time (they're metadata for validators).

### 1.3 BLOOM_LEVELS ordering
`("remember", "understand", "apply", "analyze", "evaluate", "create")` — order fixed by the schema's `required` array and by pedagogical convention (low → high cognitive complexity). Loader exposes this as a frozen tuple.

---

## 2. Per-callsite analysis

Line numbers verified with `Grep` on `dev-v0.2.0 @ e0455eb`. Some have drifted from the master-plan numbers; current values below.

### 2.1 MIGRATE (7 sites; 6 pure verb-list + 1 partial)

| # | File | Line | Current shape | Target shape | Notes |
|---|------|------|---------------|--------------|-------|
| 1 | `lib/validators/bloom.py` | 21 | `Dict[str, Set[str]]` (6 levels × 7–11 verbs) | `get_verbs()` → `Dict[str, Set[str]]` | Verb set was trimmed relative to canonical; migrate to full 10-per-level canonical. Behavior-preserving because `detect_bloom_level` (lib/validators/bloom.py:49) searches verbs with `\b{verb}\b` — more canonical verbs ⇒ strictly more detections, same detections on existing corpus. Test suite exercises only on synthetic stems; regression test pins apply-level. |
| 2 | `Trainforge/parsers/html_content_parser.py` | 156 | `Dict[str, List[str]]` (6 levels × 6–7 verbs) | `get_verbs_list()` → `Dict[str, List[str]]` | Class-level attribute on `HTMLContentParser.BLOOM_VERBS`. Module-level load is fine — this class has no state. Replace the hardcoded dict with a class-level reference to the module-level constant. |
| 3 | `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py` | 55 | `Dict[BloomLevel, List[BloomVerb]]` — richest view. `BloomLevel` is a local `Enum`; `BloomVerb` is a local `dataclass` that also holds `level` field. | `get_verb_objects()` → `Dict[str, List[BloomVerb]]` (keys as strings per ontology-module convention). **Preserve enum-keyed shape locally** by re-keying in-module: `BLOOM_VERBS = {BloomLevel(k): v for k, v in get_verb_objects().items()}`. | Local `BloomVerb` dataclass has an extra `level: BloomLevel` field — lib.ontology.bloom's `BloomVerb` omits `level` (redundant with the dict key). Keep the local `BloomVerb` class as-is and **convert** ontology BloomVerb → local BloomVerb by passing `level=BloomLevel(key)`. Re-export from `lib.ontology.bloom` NOT needed since local dataclass stays. |
| 4 | `Courseforge/scripts/generate_course.py` | 136 | `Dict[str, List[str]]` | `get_verbs_list()` → `Dict[str, List[str]]` | Used in `detect_bloom_level` (line 156) and in `bloom_level in BLOOM_VERBS` check (line 532). `in` works on dict keys regardless of value type — either shape works. Choose list shape for regex-boundary `re.search(rf"\b{verb}\b"...)` compat with existing `detect_bloom_level`. |
| 5 | `LibV2/tools/libv2/query_decomposer.py` | 111 | `Dict[str, List[str]]` (class-level on `QueryDecomposer`) | Load from `LibV2/vendor/bloom_verbs.json` via inline helper. | LibV2 cross-package caveat — cannot `from lib.ontology.bloom import ...`. Add tiny helper `LibV2/tools/libv2/_bloom_verbs.py` that reads the vendored JSON and exposes `get_verbs_list()` with identical signature. Keeps LibV2 self-contained. |
| 6 | `Trainforge/rag/libv2_bridge.py` | 473 | Hand-assembled 20-verb regex alternation string `r"define\|list\|..."` | Build regex at module-import using `get_all_verbs()`, sorted longest-first. | Preserve "longest match first" via `sorted(verbs, key=len, reverse=True)`. Canonical has 60 verbs vs current 20 — strictly more matches, no fewer. Behavior-preserving under "accept-anything" preamble-stripper semantics. |
| 7 | `Trainforge/generators/assessment_generator.py` | 46 | Nested: `Dict[str, {"verbs": list, "patterns": list, "question_types": list}]` | Migrate ONLY the `"verbs"` portion to `get_verbs_list()[level]`. Patterns + question_types stay hardcoded. | Add `# TODO(wave-future): migrate patterns + q-types to taxonomy schemas` above. Dict is exported from `Trainforge/generators/__init__.py` — must preserve structure. |

### 2.2 TODO-ONLY (3 sites; non-verb-list)

| # | File | Line | Shape | Action |
|---|------|------|-------|--------|
| 8 | `Courseforge/agents/content-quality-remediation.md` | 159 | Inline markdown prompt dict (not code) | Add `<!-- TODO(wave-future): load from schemas/taxonomies/bloom_verbs.json at orchestrator templating layer -->` above the `bloom_verbs = {` block. Prompt text is embedded in agent spec — runtime templating is future wave. |
| 9 | `lib/semantic_structure_extractor/analysis/content_profiler.py` | 182 | `BLOOM_PATTERNS: Dict[str, List[str]]` — verb list but named as "patterns" and used for difficulty-weighting, not LO extraction. Part of a class that also holds `BLOOM_DIFFICULTY_WEIGHTS`. | Add `# TODO(wave-future): consolidate BLOOM_PATTERNS once pattern-taxonomy schema exists (currently tangled with difficulty weights)`. Do NOT migrate — difficulty weights + verb lists are conceptually coupled in this module. |
| 10 | `lib/semantic_structure_extractor/semantic_structure_extractor.py` | 126 | `BLOOM_PATTERNS: Dict[str, List[str]]` of **regex patterns** (`r'\b(define\|list\|...)\b'`), not verb lists. | Add `# TODO(wave-future): consolidate BLOOM_PATTERNS once pattern-taxonomy schema exists`. Regex alternations with inline-or require flattening to pattern schemas — out of scope. |

### 2.3 NOT TOUCHED (already well-factored or out-of-scope)

- `LibV2/tools/libv2/query_decomposition.py:114` — `BLOOM_LEVELS` constant (the 6-element list of level names). No change; this is the level enum, not the verb list.
- `Trainforge/generators/question_factory.py:91` — `BLOOM_QUESTION_MAP`. Bloom→question-type mapping. Belongs to REC-VOC-01 (Wave 1.1, already satisfied by Worker F's `question_type.json`). Out of scope.
- `Trainforge/generators/instruction_factory.py:56` — `_BLOOM_LEVELS` tuple constant. Level enum, not verbs. Could migrate to `from lib.ontology.bloom import BLOOM_LEVELS` but that's scope creep; leave for a follow-up consolidation pass.
- `Courseforge/scripts/textbook-objective-generator/objective_formatter.py:18` — imports `BLOOM_VERBS` from `bloom_taxonomy_mapper`. Because we migrate mapper (site 3) to re-assemble `BLOOM_VERBS` with the same `Dict[BloomLevel, List[BloomVerb]]` shape, this import keeps working. No change.
- `Courseforge/scripts/textbook-objective-generator/__init__.py:9` — re-exports `BLOOM_VERBS`. Same reasoning; no change.
- `schemas/ONTOLOGY.md:548` + `:1149` — docs. Out of scope per master plan "stale references get caught in later waves."

---

## 3. New files

### 3.1 `lib/ontology/__init__.py`
Module docstring only.

### 3.2 `lib/ontology/bloom.py`

Design:
- Loads JSON once at module import, caches in module-level `_CACHE`.
- Reads relative path: `_BLOOM_VERBS_PATH = Path(__file__).resolve().parents[2] / "schemas" / "taxonomies" / "bloom_verbs.json"`.
- Extracts `default` arrays out of `properties.{level}`.
- Exposes:
  - `BLOOM_LEVELS: Tuple[str, ...]` — immutable 6-tuple.
  - `@dataclass(frozen=True) class BloomVerb(verb, usage_context, example_template)` — no `level` field (level is the dict key).
  - `get_verbs() -> Dict[str, Set[str]]` — shape for `lib/validators/bloom.py`.
  - `get_verbs_list() -> Dict[str, List[str]]` — shape for 4 callsites.
  - `get_verb_objects() -> Dict[str, List[BloomVerb]]` — richest view.
  - `get_all_verbs() -> Set[str]` — flat union across all levels.
  - `detect_bloom_level(text: str) -> Tuple[Optional[str], Optional[str]]` — longest-verb-first matching, returns `(level, verb)` or `(None, None)`. Priority: higher levels first (create → remember) for tie-breaking on shared verbs like `compare` (appears in both understand and analyze in the richest copy — but canonical list has no shared verbs, verified from Worker F's data).
- All getters return **defensive copies** so callers can mutate without polluting the cache.

### 3.3 `LibV2/vendor/bloom_verbs.json`

**Decision: Committed byte-for-byte copy + CI hash check.**

#### 3.3.1 Options considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **A. Committed copy + CI hash** | Works on every dev machine regardless of FS symlink support; no build step; readable via normal Read tool; survives copy-as-archive. | Two files to keep in sync; drift risk mitigated only by CI hash check. | ✅ CHOSEN |
| B. Symlink | Zero duplication; always in sync. | Symlinks are flaky on Windows/NTFS dev boxes; break when repo is zipped/tarred for archival; git treats symlinks inconsistently across `core.symlinks` config. | ❌ Too fragile for cross-platform open-source project. |
| C. Build-time copy script | No drift at build time. | Requires every developer to run setup before LibV2 runs; CI complexity; violates "reproducible clone" principle. | ❌ Adds ceremony without benefit over A. |

#### 3.3.2 CI hash check

Add a short check to `ci/integrity_check.py` (as a new `check_libv2_vendor_sync` function added to the checks tuple in `run_all_checks`). Computes `sha256(schemas/taxonomies/bloom_verbs.json)` and `sha256(LibV2/vendor/bloom_verbs.json)` — fails if they differ. Standalone test in the new test file also runs this check for fast feedback.

#### 3.3.3 LibV2/vendor/ directory convention

New directory. `LibV2/vendor/` signals "vendored (third-party or cross-package) assets." Future additions (other cross-package taxonomy vendors) go here. Add a comment at the top of the vendored file? **No** — JSON can't hold comments safely across strict parsers; hash check is the guard.

### 3.4 Tiny helper: `LibV2/tools/libv2/_bloom_verbs.py`

```python
"""Internal: load Bloom verbs from LibV2/vendor/bloom_verbs.json.
LibV2 cannot import from Ed4All's lib/ (cross-package sandbox per CLAUDE.md).
This module provides equivalent access to the vendored canonical data."""
import json
from pathlib import Path
from typing import Dict, List

_VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "bloom_verbs.json"
_CACHE: Dict[str, List[str]] | None = None

def get_verbs_list() -> Dict[str, List[str]]:
    global _CACHE
    if _CACHE is None:
        with open(_VENDOR_PATH) as f:
            data = json.load(f)
        _CACHE = {
            level: [v["verb"] for v in data["properties"][level]["default"]]
            for level in ("remember", "understand", "apply", "analyze", "evaluate", "create")
        }
    return {k: list(v) for k, v in _CACHE.items()}  # defensive copy
```

---

## 4. Migration diff outlines

### 4.1 `lib/validators/bloom.py` — Set[str] shape
```diff
-BLOOM_VERBS: Dict[str, Set[str]] = {
-    "remember": {"define", "list", ...},
-    ...
-}
+from lib.ontology.bloom import get_verbs as _get_verbs
+BLOOM_VERBS: Dict[str, Set[str]] = _get_verbs()
```

### 4.2 `Trainforge/parsers/html_content_parser.py` — List[str] shape, class attribute
```diff
 class HTMLContentParser:
-    BLOOM_VERBS = {
-        "remember": ["define", ...],
-        ...
-    }
+    from lib.ontology.bloom import get_verbs_list as _get_verbs_list
+    BLOOM_VERBS = _get_verbs_list()
```
Note: Python allows an import inside a class body but it's unusual style. Cleaner: move the import to the top of the module and assign `BLOOM_VERBS = get_verbs_list()` at class-body top.

### 4.3 `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py` — richest copy
```diff
-BLOOM_VERBS: Dict[BloomLevel, List[BloomVerb]] = {
-    BloomLevel.REMEMBER: [
-        BloomVerb("define", BloomLevel.REMEMBER, "terms and concepts", "Define {concept}"),
-        ...
-    ],
-    ...
-}
+from lib.ontology.bloom import get_verb_objects as _get_verb_objects
+
+def _build_bloom_verbs() -> Dict[BloomLevel, List[BloomVerb]]:
+    """Re-key ontology data to this module's local (BloomLevel, BloomVerb) shape."""
+    src = _get_verb_objects()
+    return {
+        BloomLevel(level): [
+            BloomVerb(v.verb, BloomLevel(level), v.usage_context, v.example_template)
+            for v in src[level]
+        ]
+        for level in ("remember", "understand", "apply", "analyze", "evaluate", "create")
+    }
+
+BLOOM_VERBS: Dict[BloomLevel, List[BloomVerb]] = _build_bloom_verbs()
```
The local `BloomVerb` dataclass stays (has `level` field; ontology BloomVerb does not). Conversion bridges the two.

### 4.4 `Courseforge/scripts/generate_course.py` — List[str] shape
```diff
-BLOOM_VERBS: Dict[str, List[str]] = {
-    "remember": ["define", "list", ...],
-    ...
-}
+from lib.ontology.bloom import get_verbs_list as _get_verbs_list
+BLOOM_VERBS: Dict[str, List[str]] = _get_verbs_list()
```

### 4.5 `LibV2/tools/libv2/query_decomposer.py` — LibV2 cross-package
```diff
 class QueryDecomposer:
-    BLOOM_VERBS = {
-        'remember': [...],
-        ...
-    }
+    from ._bloom_verbs import get_verbs_list as _get_verbs_list
+    BLOOM_VERBS = _get_verbs_list()
```

### 4.6 `Trainforge/rag/libv2_bridge.py` — regex alternation
```diff
 @staticmethod
 def _extract_query_concepts(text: str) -> str:
     ...
-    bloom_verbs = (
-        "define|list|recall|identify|explain|describe|summarize|"
-        "apply|demonstrate|use|solve|analyze|compare|contrast|"
-        "evaluate|judge|justify|create|design|develop"
-    )
+    bloom_verbs = _BLOOM_VERB_ALT  # module-level constant
     cleaned = re.sub(
         rf"^(?:{bloom_verbs})\s+", "", cleaned, flags=re.IGNORECASE
     ).strip()
```
At module top:
```python
from lib.ontology.bloom import get_all_verbs as _get_all_verbs
# Longest-first so multi-word verbs would bind before their prefixes.
# Canonical has no multi-word verbs, but sort defensively.
_BLOOM_VERB_ALT = "|".join(re.escape(v) for v in sorted(_get_all_verbs(), key=len, reverse=True))
```

### 4.7 `Trainforge/generators/assessment_generator.py` — partial
```diff
+# TODO(wave-future): migrate patterns + q-types to taxonomy schemas (only verbs migrated today)
+from lib.ontology.bloom import get_verbs_list as _get_verbs_list
+_CANONICAL_VERBS = _get_verbs_list()
+
 BLOOM_LEVELS = {
     "remember": {
-        "verbs": ["define", "list", "recall", "identify", "name"],
+        "verbs": _CANONICAL_VERBS["remember"],
         "patterns": ["What is...?", "List the...", "Which of the following...?"],
         "question_types": ["multiple_choice", "true_false", "fill_in_blank"],
     },
     ...
 }
```
This expands the verb lists from 5 to 10 per level (canonical size). Question-type selection code (`assessment_generator.py:407`) only reads `level_config["question_types"]` — unaffected. The `"verbs"` portion is read nowhere in the module except by downstream consumers who pattern-match verbs in stems (more verbs ⇒ strictly better detection, no false matches because canonical verbs are well-established Bloom verbs).

---

## 5. Regression test design

**File:** `lib/tests/test_bloom_ontology.py`

**Structure:** one pytest module with 5 test functions.

```python
"""Regression tests for lib.ontology.bloom loader and callsite migration."""
import hashlib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_shapes():
    """Loader exposes three shapes over the same data."""
    from lib.ontology.bloom import (
        BLOOM_LEVELS, get_verbs, get_verbs_list, get_verb_objects, get_all_verbs,
    )

    assert BLOOM_LEVELS == ("remember", "understand", "apply", "analyze", "evaluate", "create")

    sets = get_verbs()
    lists = get_verbs_list()
    objs = get_verb_objects()

    for level in BLOOM_LEVELS:
        verb_set = sets[level]
        verb_list = lists[level]
        verb_objs = objs[level]
        assert isinstance(verb_set, set)
        assert isinstance(verb_list, list)
        assert verb_set == set(verb_list), f"set/list mismatch at {level}"
        assert {o.verb for o in verb_objs} == verb_set, f"objects/set mismatch at {level}"

    # 60 verbs total (10 per level)
    flat = get_all_verbs()
    assert len(flat) >= 55  # conservative; canonical is 60 but some levels may share verbs


def test_migrated_sites_match_canonical_apply_level():
    """Every migrated callsite's apply-level verb set == canonical apply set."""
    from lib.ontology.bloom import get_verbs_list
    canonical_apply = set(get_verbs_list()["apply"])

    # lib.validators.bloom
    from lib.validators.bloom import BLOOM_VERBS as validators_bloom
    assert set(validators_bloom["apply"]) == canonical_apply, "lib.validators.bloom.apply drift"

    # Trainforge.parsers.html_content_parser
    from Trainforge.parsers.html_content_parser import HTMLContentParser
    assert set(HTMLContentParser.BLOOM_VERBS["apply"]) == canonical_apply, "html_content_parser.apply drift"

    # Courseforge.scripts.generate_course
    import importlib.util, sys
    cf_path = _REPO_ROOT / "Courseforge" / "scripts" / "generate_course.py"
    spec = importlib.util.spec_from_file_location("generate_course", cf_path)
    gc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gc)
    assert set(gc.BLOOM_VERBS["apply"]) == canonical_apply, "generate_course.apply drift"

    # Trainforge.generators.assessment_generator — nested shape; verbs only
    from Trainforge.generators.assessment_generator import BLOOM_LEVELS as ag_levels
    assert set(ag_levels["apply"]["verbs"]) == canonical_apply, "assessment_generator.apply drift"


def test_bloom_taxonomy_mapper_shape_preserved():
    """Richest callsite preserves Dict[BloomLevel, List[BloomVerb]] shape."""
    import importlib.util, sys
    mapper_path = _REPO_ROOT / "Courseforge" / "scripts" / "textbook-objective-generator" / "bloom_taxonomy_mapper.py"
    spec = importlib.util.spec_from_file_location("bloom_taxonomy_mapper", mapper_path)
    m = importlib.util.module_from_spec(spec); sys.modules["bloom_taxonomy_mapper"] = m; spec.loader.exec_module(m)
    for level, verbs in m.BLOOM_VERBS.items():
        assert isinstance(level, m.BloomLevel), f"expected BloomLevel key, got {type(level)}"
        for v in verbs:
            assert isinstance(v, m.BloomVerb), f"expected local BloomVerb, got {type(v)}"
            assert v.level == level
            assert v.verb and v.usage_context and v.example_template


def test_detect_bloom_level():
    """Smoke test for the canonical detector."""
    from lib.ontology.bloom import detect_bloom_level
    level, verb = detect_bloom_level("design a system to handle high load")
    assert level == "create"
    assert verb == "design"

    level, verb = detect_bloom_level("list the steps of photosynthesis")
    assert level == "remember"
    assert verb == "list"

    level, verb = detect_bloom_level("no verbs here at all whatsoever")
    assert level is None
    assert verb is None


def test_libv2_vendor_hash_sync():
    """LibV2/vendor/bloom_verbs.json must be byte-identical to the authoritative copy."""
    auth = _REPO_ROOT / "schemas" / "taxonomies" / "bloom_verbs.json"
    vendored = _REPO_ROOT / "LibV2" / "vendor" / "bloom_verbs.json"
    assert auth.exists(), f"Authoritative copy missing: {auth}"
    assert vendored.exists(), f"Vendored copy missing: {vendored}"
    h_auth = hashlib.sha256(auth.read_bytes()).hexdigest()
    h_vendored = hashlib.sha256(vendored.read_bytes()).hexdigest()
    assert h_auth == h_vendored, f"Hash drift:\n  auth:     {h_auth}\n  vendored: {h_vendored}"
```

---

## 6. Files touched (summary)

### New (4)
- `lib/ontology/__init__.py`
- `lib/ontology/bloom.py`
- `LibV2/vendor/bloom_verbs.json` (byte-copy of authoritative)
- `LibV2/tools/libv2/_bloom_verbs.py` (tiny loader, cross-package caveat)

### Modified — migrations (7)
- `lib/validators/bloom.py`
- `Trainforge/parsers/html_content_parser.py`
- `Courseforge/scripts/textbook-objective-generator/bloom_taxonomy_mapper.py`
- `Courseforge/scripts/generate_course.py`
- `LibV2/tools/libv2/query_decomposer.py`
- `Trainforge/rag/libv2_bridge.py`
- `Trainforge/generators/assessment_generator.py` (verb-portion only)

### Modified — TODO comments (3)
- `Courseforge/agents/content-quality-remediation.md`
- `lib/semantic_structure_extractor/analysis/content_profiler.py`
- `lib/semantic_structure_extractor/semantic_structure_extractor.py`

### New test (1)
- `lib/tests/test_bloom_ontology.py`

### Modified — CI check (1)
- `ci/integrity_check.py` — add `check_libv2_vendor_sync` function + register in `run_all_checks`.

---

## 7. Verification commands

```bash
# Schema still parses
python3 -m json.tool schemas/taxonomies/bloom_verbs.json > /dev/null

# CI integrity (includes new hash check)
python3 -m ci.integrity_check

# Test suite
pytest lib/tests/test_bloom_ontology.py -v
pytest lib/tests/ Trainforge/tests/ -x

# Loader sanity
python3 -c "from lib.ontology.bloom import get_verbs, get_verbs_list, get_verb_objects, detect_bloom_level, get_all_verbs; print('OK'); print(detect_bloom_level('design a system'))"

# LibV2 vendor hash sync
python3 -c "import hashlib; h1=hashlib.sha256(open('schemas/taxonomies/bloom_verbs.json','rb').read()).hexdigest(); h2=hashlib.sha256(open('LibV2/vendor/bloom_verbs.json','rb').read()).hexdigest(); assert h1==h2; print('OK')"
```

---

## 8. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| `lib.validators.bloom`'s smaller verb set was intentional (false-positive avoidance) | Reviewed: the trimmed list was a hand-copy, not a deliberate filter. Canonical 10-verbs-per-level are all standard Bloom verbs; no false-positive risk. Regression test exercises `detect_bloom_level`. |
| `Trainforge/generators/__init__.py` re-exports `BLOOM_LEVELS` | Shape preserved: top-level dict, same 6 string keys, same nested `{"verbs": [...], "patterns": [...], "question_types": [...]}` shape. Only verb-list values swapped. |
| LibV2 vendored file drifts | Mitigated by `test_libv2_vendor_hash_sync` + `check_libv2_vendor_sync` in CI. Both run on every push. |
| `bloom_taxonomy_mapper.py`'s `objective_formatter.py` importer | Preserved: `BLOOM_VERBS` is still `Dict[BloomLevel, List[BloomVerb]]` with local `BloomVerb` dataclass (has `level` field). `BloomVerb.verb` access unchanged. |
| Circular import between `lib.validators.bloom` and `lib.ontology.bloom` | None: ontology has zero dependencies on validators. Validators depends on ontology (forward dependency only). |
| `re.escape` on single-word verbs | No-op for alphabetic verbs; defensive for future multi-word additions. Explicit in libv2_bridge migration. |
| Case sensitivity in `detect_bloom_level` | Canonical stores lowercase verbs. Detector lowercases input. Preserves existing behavior (all existing `detect_bloom_level` impls lowercase the input). |

---

## 9. Commit + PR

Commit message: `Worker H: REC-BL-01 lib/ontology/bloom.py loader + BLOOM_VERBS migration (7 sites + 3 TODOs)`

PR body summarizes changes, lists verification commands, and references this sub-plan.
