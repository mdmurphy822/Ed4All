# Worker K Sub-Plan — Wave 2 REC-VOC-02 `teaching_role` emit + consume alignment

**Branch:** `worker-k/wave2-teaching-role`
**Base:** `dev-v0.2.0` @ `4159576` (after Wave 1 merged: Workers F/G/H/I)
**Depends on:** `schemas/taxonomies/teaching_role.json` (Worker F, Wave 1.1)
**Parallels:** Worker J (`_build_page_metadata`, `generate_week`, `generate_course`, `Trainforge/process_course.py`), Worker L (packager + workflows config). No file overlap.

---

## 1. Authoritative `teaching_role.json` mapping (verified)

Worker F's `schemas/taxonomies/teaching_role.json` publishes the canonical six-valued enum plus an `x-component-mapping` extension keyword that encodes the deterministic `(component, purpose) → teaching_role` mapping.

### 1.1 Canonical roles (`$defs.TeachingRole.enum`)
```
introduce, elaborate, reinforce, assess, transfer, synthesize
```
Must match `Trainforge/align_chunks.py:33 VALID_ROLES`.

### 1.2 `x-component-mapping` entries (all three)

| component     | purpose                | teaching_role | source                                  |
|---------------|------------------------|---------------|-----------------------------------------|
| `flip-card`   | `term-definition`      | `introduce`   | `Courseforge/scripts/generate_course.py:345` |
| `self-check`  | `formative-assessment` | `assess`      | `Courseforge/scripts/generate_course.py:374` |
| `activity`    | `practice`             | `transfer`    | `Courseforge/scripts/generate_course.py:487` |

These three are the full universe of `data-cf-component`/`data-cf-purpose` pairs currently emitted by `generate_course.py`. The mapper returns `None` for any other input, which triggers the caller's fallback path (JSON-LD or LLM). Future component types can be added in the schema file without code changes — the loader reads the array at runtime.

### 1.3 Shape notes
- The schema's `$ref` at the root points at `TeachingRole`, so the schema validates a string value (the enum). The mapping metadata lives under the `x-` prefix which is reserved for schema extensions and ignored by standard JSON-Schema validators. Safe to consume at load time.

---

## 2. New file: `lib/ontology/teaching_roles.py`

### 2.1 Design (mirrors `lib/ontology/bloom.py`)
- Reads `schemas/taxonomies/teaching_role.json` once at module import, caches with `@lru_cache(maxsize=1)`.
- Path: `Path(__file__).resolve().parents[2] / "schemas" / "taxonomies" / "teaching_role.json"`.
- Public surface:
  - `TEACHING_ROLES: Tuple[str, ...]` — immutable 6-tuple in schema-declared order.
  - `load_teaching_roles() -> Dict` — raw schema dict (for advanced callers / tests).
  - `get_valid_roles() -> Set[str]` — canonical set.
  - `map_role(component: Optional[str], purpose: Optional[str]) -> Optional[str]` — returns mapped role or `None` when no match. Handles `None`/empty inputs without raising.

### 2.2 `map_role` semantics
- Exact match on both `component` and `purpose` required.
- Case-sensitive (all schema values are lowercase kebab-case; all emit sites use lowercase kebab-case).
- Returns `None` when:
  - component is None or empty,
  - purpose is None or empty,
  - the `(component, purpose)` pair is not declared in `x-component-mapping`.
- Built as a frozen `Dict[Tuple[str, str], str]` lookup keyed by the pair. One-time construction at import via `@lru_cache`.

### 2.3 Import-time fail-fast
If the schema file is missing or malformed: raise `FileNotFoundError` / `ValueError` with an actionable message (mirrors `bloom.py` pattern). Downstream failures should surface at module load, not at the first `map_role` call.

---

## 3. New file: `lib/tests/test_teaching_roles.py`

Four tests per the master plan:

1. **`test_map_known_pairs`** — every declared mapping returns the expected role:
   - `map_role("flip-card", "term-definition") == "introduce"`
   - `map_role("self-check", "formative-assessment") == "assess"`
   - `map_role("activity", "practice") == "transfer"`
2. **`test_map_unknown_returns_none`** — unmapped pairs return None:
   - `map_role("accordion", "progressive-disclosure") is None`
   - `map_role("flip-card", "bogus-purpose") is None`
   - `map_role("bogus-component", "term-definition") is None`
3. **`test_map_partial_returns_none`** — None/empty component or purpose returns None:
   - `map_role(None, "term-definition") is None`
   - `map_role("flip-card", None) is None`
   - `map_role("", "term-definition") is None`
   - `map_role("flip-card", "") is None`
4. **`test_valid_roles_is_six_values`** — canonical set matches Trainforge:
   - Import `VALID_ROLES` from `Trainforge.align_chunks`; assert `get_valid_roles() == VALID_ROLES` (both equal the 6-element canonical set).

Plus one dynamic cross-check:
5. **`test_mapping_covers_declared_emit_sites`** — iterate the schema's `x-component-mapping` and re-verify every entry round-trips through `map_role`.

---

## 4. Modify `Courseforge/scripts/generate_course.py`

All three emit sites receive one new `data-cf-teaching-role="<role>"` attribute. Source of truth is `map_role(component, purpose)` — no hardcoded strings.

Import at module top (after existing bloom import):
```python
from lib.ontology.teaching_roles import map_role as _map_teaching_role  # noqa: E402
```

### 4.1 `_render_flip_cards` (L339; emit at L345–349)

**Old:**
```python
cards.append(f"""
  <div class="flip-card" tabindex="0" role="button" aria-label="Flip card: {front}"
       data-cf-component="flip-card" data-cf-purpose="term-definition"
       data-cf-term="{term_slug}">
```

**New:**
```python
_FC_ROLE = _map_teaching_role("flip-card", "term-definition") or ""
_fc_role_attr = f' data-cf-teaching-role="{_FC_ROLE}"' if _FC_ROLE else ""
cards.append(f"""
  <div class="flip-card" tabindex="0" role="button" aria-label="Flip card: {front}"
       data-cf-component="flip-card" data-cf-purpose="term-definition"{_fc_role_attr}
       data-cf-term="{term_slug}">
```

Rationale: `map_role` returns `None` only if the schema is malformed — we guard emit with `if _FC_ROLE` so an upstream schema bug doesn't leak empty `data-cf-teaching-role=""` to HTML.

### 4.2 `_render_self_check` (L358; emit at L376–379)

**Old `sc_attrs` construction:**
```python
sc_attrs = (
    f' data-cf-component="self-check" data-cf-purpose="formative-assessment"'
    f' data-cf-bloom-level="{bloom}"'
)
```

**New:**
```python
_sc_role = _map_teaching_role("self-check", "formative-assessment")
sc_attrs = (
    f' data-cf-component="self-check" data-cf-purpose="formative-assessment"'
    + (f' data-cf-teaching-role="{_sc_role}"' if _sc_role else "")
    + f' data-cf-bloom-level="{bloom}"'
)
```

### 4.3 `_render_activities` (L483; emit at L489–492)

**Old `act_attrs` construction:**
```python
act_attrs = (
    f' data-cf-component="activity" data-cf-purpose="practice"'
    f' data-cf-bloom-level="{bloom}"'
)
```

**New:**
```python
_act_role = _map_teaching_role("activity", "practice")
act_attrs = (
    f' data-cf-component="activity" data-cf-purpose="practice"'
    + (f' data-cf-teaching-role="{_act_role}"' if _act_role else "")
    + f' data-cf-bloom-level="{bloom}"'
)
```

### 4.4 `_build_sections_metadata` (L552)

`_build_sections_metadata` only sees content sections (not self-check or activity pages — those are rendered separately, one per page). Within a section, only flip-cards have a mappable role (via `("flip-card", "term-definition")`). So the role-set for a section is `{"introduce"}` when flip-cards are present, otherwise empty.

**Design:** add a helper `_collect_section_roles(section)` that walks components and yields mapped roles. Emit as a sorted list when non-empty.

**Old (L552–L571):**
```python
def _build_sections_metadata(sections: List[Dict]) -> List[Dict[str, Any]]:
    """Build structured section metadata for JSON-LD."""
    result = []
    for section in sections:
        content_type = section.get("content_type") or _infer_content_type(section)
        entry: Dict[str, Any] = {
            "heading": section["heading"],
            "contentType": content_type,
        }
        # Key terms from flip_cards
        if section.get("flip_cards"):
            entry["keyTerms"] = [
                {"term": t["term"], "definition": t["definition"]}
                for t in section["flip_cards"]
            ]
        bloom_range = section.get("bloom_range")
        if bloom_range:
            entry["bloomRange"] = [bloom_range] if isinstance(bloom_range, str) else bloom_range
        result.append(entry)
    return result
```

**New (inserts one block between `keyTerms` and `bloomRange`):**
```python
def _collect_section_roles(section: Dict) -> List[str]:
    """Collect deterministic teachingRole values for components inside a section.

    Walks the section's flip_cards / self_check / activities children and
    maps each (component, purpose) pair via lib.ontology.teaching_roles.
    Returns a sorted list (stable for diff-friendly JSON-LD output).
    """
    roles: Set[str] = set()
    if section.get("flip_cards"):
        r = _map_teaching_role("flip-card", "term-definition")
        if r:
            roles.add(r)
    # Future-compat: sections CURRENTLY don't carry self_check/activities
    # inline (they live at week scope). If that invariant changes, the
    # mapper extension below handles it deterministically.
    for q in section.get("self_check", []) or []:
        r = _map_teaching_role("self-check", "formative-assessment")
        if r:
            roles.add(r)
    for a in section.get("activities", []) or []:
        r = _map_teaching_role("activity", "practice")
        if r:
            roles.add(r)
    return sorted(roles)


def _build_sections_metadata(sections: List[Dict]) -> List[Dict[str, Any]]:
    """Build structured section metadata for JSON-LD."""
    result = []
    for section in sections:
        content_type = section.get("content_type") or _infer_content_type(section)
        entry: Dict[str, Any] = {
            "heading": section["heading"],
            "contentType": content_type,
        }
        # Key terms from flip_cards
        if section.get("flip_cards"):
            entry["keyTerms"] = [
                {"term": t["term"], "definition": t["definition"]}
                for t in section["flip_cards"]
            ]
        # Teaching roles collected from tagged components inside the section
        teaching_roles = _collect_section_roles(section)
        if teaching_roles:
            entry["teachingRole"] = teaching_roles
        bloom_range = section.get("bloom_range")
        if bloom_range:
            entry["bloomRange"] = [bloom_range] if isinstance(bloom_range, str) else bloom_range
        result.append(entry)
    return result
```

Note: `Set` will need to be added to the `typing` imports near the top of the file (already imports `List, Dict, Any, Optional, Tuple` — add `Set`).

### 4.5 NOT touching (respecting J's scope)
- `_build_page_metadata` (L574) — Worker J's surface.
- `generate_week` (L601) — Worker J's surface.
- `generate_course` (L771) — Worker J's surface.
- `_build_objectives_metadata` (L515) — no teaching_role fit here.

---

## 5. Modify `Trainforge/parsers/html_content_parser.py`

### 5.1 Extend `ContentSection` dataclass (L28–L37)

Add one optional field:
```python
teaching_role: Optional[str] = None  # from data-cf-teaching-role
```

### 5.2 Extend heading-attr extraction in `_extract_sections` (L268–L314)

Currently the section-attr scan reads `data-cf-content-type` and `data-cf-key-terms` off the `<h*>` tag. `data-cf-teaching-role` is not on the heading — it's on the inner flip-card / self-check / activity elements. Adding section-level teaching_role to `ContentSection` requires a per-section HTML scan for the attribute.

**Strategy:** after existing attr extraction, scan the section's inner HTML for `data-cf-teaching-role="..."` tokens. Collect a set; if exactly one value is present, set `teaching_role` on the section. If multiple distinct values are present, leave as None (ambiguous — consumer should look at JSON-LD `teachingRole` array instead).

```python
# Extract teaching role(s) from tagged components inside the section.
tr_matches = re.findall(r'data-cf-teaching-role="([^"]*)"', section_html)
distinct = {r for r in tr_matches if r}
teaching_role = next(iter(distinct)) if len(distinct) == 1 else None
```

Surface on the dataclass:
```python
sections.append(ContentSection(
    ...,
    teaching_role=teaching_role,
))
```

### 5.3 Why not extract at chunk-level here?

Chunks are built downstream (`Trainforge/process_course.py`, `Trainforge/align_chunks.py`). `align_chunks.classify_teaching_role` needs the attribute per-chunk. The simplest, least-invasive surface is:
1. This parser exposes `teaching_role` on `ContentSection`.
2. The chunker (section → chunk) propagates the field onto the chunk dict as `teaching_role_attr`.

However, fully wiring the chunker is outside Worker K's scope without touching process_course.py (which is J's file). The consume-side precedence chain in `align_chunks.py` can also parse the HTML directly via the existing chunk `source` dict if the raw HTML is accessible. **Fallback plan:** if chunker doesn't already expose `teaching_role_attr`, Worker K adds a lightweight per-chunk scan at the top of `classify_teaching_role` that looks at `chunk.get("teaching_role_attr")` first, then falls back to a regex scan of `chunk.get("text", "")` for `data-cf-teaching-role="..."` when the chunk was built before this field existed. This keeps the attr-priority path deterministic regardless of chunker version.

**Scope decision:** Worker K surfaces `teaching_role` on `ContentSection` (used by parsers for section-level metadata), and updates `align_chunks.classify_teaching_role` to consume `chunk.get("teaching_role_attr")` / `chunk.get("source", {}).get("teaching_role")` if present. Wiring the field from ContentSection into the actual chunk dicts in process_course.py belongs to a follow-up worker (or J when he touches that file anyway) — documented as TODO here.

---

## 6. Modify `Trainforge/align_chunks.py` (L465–L582)

### 6.1 New precedence chain

Add a per-chunk preflight before the existing heuristic check in `classify_teaching_roles`:

```python
for chunk in chunks:
    # 1. Deterministic: data-cf-teaching-role attribute surfaced by the parser
    attr_role = chunk.get("teaching_role_attr")
    if attr_role and attr_role in VALID_ROLES:
        chunk["teaching_role"] = attr_role
        chunk["teaching_role_source"] = "attr"
        if verbose:
            print(f"  {chunk['id']}: role={attr_role} (deterministic:attr)")
        continue

    # 2. Deterministic: JSON-LD section teachingRole (unambiguous single-value case)
    section_roles = (chunk.get("source", {}).get("section_teaching_roles") or [])
    if len(section_roles) == 1 and section_roles[0] in VALID_ROLES:
        chunk["teaching_role"] = section_roles[0]
        chunk["teaching_role_source"] = "jsonld"
        if verbose:
            print(f"  {chunk['id']}: role={section_roles[0]} (deterministic:jsonld)")
        continue

    # 3. Fall through to existing heuristic / LLM path
    role = _heuristic_role(chunk)
    if role:
        chunk["teaching_role"] = role
        chunk["teaching_role_source"] = "heuristic"
        heuristic_count += 1
        if verbose:
            print(f"  {chunk['id']}: role={role} (heuristic)")
    else:
        ambiguous_chunks.append(chunk)
```

### 6.2 Import canonical set (don't duplicate)

Currently `VALID_ROLES` is a local set. After Worker F/Worker K, `lib.ontology.teaching_roles.get_valid_roles()` exposes the same canonical set. **Keep the existing local `VALID_ROLES` constant** (it's at L33; other code in the file references it) and add a one-time assertion near the top of `classify_teaching_roles` that the two sets match — fail-closed if schema drift occurs:

```python
# Optional belt-and-suspenders: catch schema drift early.
try:
    from lib.ontology.teaching_roles import get_valid_roles as _canonical_valid_roles
    assert VALID_ROLES == _canonical_valid_roles(), (
        f"teaching_role schema drift: align_chunks.VALID_ROLES={VALID_ROLES} "
        f"vs schemas/taxonomies/teaching_role.json={_canonical_valid_roles()}"
    )
except ImportError:
    pass  # running outside the repo (e.g., standalone Trainforge install)
```

### 6.3 Preserve LLM fallback

`_classify_with_llm` (L510) stays untouched. The existing `for chunk in chunks:` loop body is wrapped with the deterministic preflight above — only chunks that don't hit the attr or JSON-LD paths continue into the heuristic/LLM pipeline.

### 6.4 Accounting

Add a `deterministic_count` counter and include it in the summary print:
```python
print(f"  Teaching roles: {deterministic_count} deterministic, "
      f"{heuristic_count} heuristic, "
      f"{llm_count or len(ambiguous_chunks)} {'LLM' if llm_provider == 'anthropic' else 'mock'}")
```

---

## 7. Regression test: `Trainforge/tests/test_teaching_role_emit.py`

Five tests per the master plan:

1. **`test_flip_card_emits_introduce`**
   - Synthesize one terms dict: `[{"term": "API", "definition": "interface"}]`.
   - Call `_render_flip_cards` directly (imported via importlib from `Courseforge/scripts/generate_course.py`).
   - Assert `data-cf-teaching-role="introduce"` in returned HTML.
   - Assert exactly one occurrence.

2. **`test_self_check_emits_assess`**
   - Synthesize one question dict with options.
   - Call `_render_self_check`.
   - Assert `data-cf-teaching-role="assess"` in output.

3. **`test_activity_emits_transfer`**
   - Synthesize one activity dict.
   - Call `_render_activities`.
   - Assert `data-cf-teaching-role="transfer"` in output.

4. **`test_section_jsonld_teaching_role_array`**
   - Synthesize sections list: one section with `flip_cards: [{...}]`, one without.
   - Call `_build_sections_metadata`.
   - Assert first entry has `teachingRole == ["introduce"]` (single-element sorted list).
   - Assert second entry has no `teachingRole` key.

5. **`test_align_chunks_prefers_deterministic`**
   - Mock chunk: `{"id": "c1", "text": "...", "teaching_role_attr": "introduce", "_position": 0}`.
   - Import `classify_teaching_roles` from `Trainforge.align_chunks`.
   - Call with `llm_provider="anthropic"` (would fail fast on anthropic import but the preflight should skip that path entirely for deterministic chunks).
   - Assert `chunk["teaching_role"] == "introduce"`.
   - Assert `chunk["teaching_role_source"] == "attr"`.

Plus two supplementary tests:

6. **`test_align_chunks_jsonld_precedence`**
   - Mock chunk without `teaching_role_attr` but with `source.section_teaching_roles == ["transfer"]`.
   - Assert precedence: role resolves from JSON-LD, source is `"jsonld"`.

7. **`test_align_chunks_ambiguous_jsonld_falls_through`**
   - Mock chunk with `source.section_teaching_roles == ["introduce", "assess"]` (multi-value).
   - Assert classifier falls through to heuristic/mock path (does NOT pick one of the jsonld roles).

### 7.1 Import strategy
`generate_course.py` is a script, not a package. Import via `importlib.util` the same way `lib/tests/test_bloom_ontology.py` does (see `_load_by_path` helper there).

---

## 8. Verification plan

```bash
# Pre-checks
python3 -m ci.integrity_check

# Unit tests
source venv/bin/activate
pytest lib/tests/test_teaching_roles.py -v
pytest Trainforge/tests/test_teaching_role_emit.py -v

# Integration: regenerate a small WCAG_201 page and grep the output
python3 Courseforge/scripts/generate_course.py \
    Courseforge/inputs/exam-objectives/SAMPLE_101_course_data.json \
    /tmp/worker_k_smoke
grep -l "data-cf-teaching-role" /tmp/worker_k_smoke/week_*/*.html
# Expect: every flip-card/self-check/activity page lists it.
```

(If WCAG_201 input JSON is unavailable in the worktree, fall back to SAMPLE_101 or any existing fixture.)

---

## 9. Commit + PR

Single commit on `worker-k/wave2-teaching-role`:

```
Worker K: REC-VOC-02 — teaching_role emit + deterministic consume
```

Target: `dev-v0.2.0`. Base: main off-limits.

---

## 10. Scope guards (from master plan)

- Do NOT modify `schemas/taxonomies/teaching_role.json` (F owns).
- Do NOT touch `_build_page_metadata` / `generate_week` / `generate_course` (J owns).
- Do NOT touch `Courseforge/scripts/package_multifile_imscc.py` or `config/workflows.yaml` (L owns).
- Do NOT remove the LLM classifier — it remains as fallback for legacy IMSCCs.
- Deterministic paths MUST execute before the LLM path — that is the whole point of REC-VOC-02.

---

## 11. Open questions / deferred

- **Chunker → chunk dict wiring:** the field `teaching_role_attr` on chunks depends on `Trainforge/process_course.py` propagating `ContentSection.teaching_role` onto each chunk. Full propagation belongs to a follow-up worker (or J since he already touches process_course.py for TAX-01). Worker K's `classify_teaching_roles` is defensive: if the field isn't present, it falls through to the heuristic/LLM path — same behavior as today. No regression risk.
- **JSON-LD `teachingRole` consumption in align_chunks:** Worker K reads from `chunk.source.section_teaching_roles`, expecting future chunker propagation. Same defensive fallthrough applies.
