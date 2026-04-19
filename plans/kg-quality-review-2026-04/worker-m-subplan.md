# Worker M Sub-Plan — Wave 3 consume-side alignment + packager cleanup

**Branch:** `worker-m/wave3-consume-alignment`
**Base:** `dev-v0.2.0` @ `befd6f6` (after Wave 2 merged: Workers J/K/L)
**Master plan:** `/home/mdmur/.claude/plans/we-have-several-branches-gentle-melody.md`

Three thematically-aligned items:
1. **REC-JSL-03** — Trainforge parses `data-cf-objective-ref` on activities/self-checks; attaches to chunk `learning_outcome_refs[]`.
2. **Case preservation (A3)** — opt-in env var `TRAINFORGE_PRESERVE_LO_CASE=true` to stop lowercasing LO refs at ingest.
3. **Packager stub inclusion** — `package_multifile_imscc.py` bundles `course_metadata.json` in the IMSCC zip when present.

---

## 1. REC-JSL-03 design

### 1.1 Emit-side anchors (verified)

`Courseforge/scripts/generate_course.py`:
- **L374–396 `_render_self_check`** — emits `<div class="self-check"...>` with `data-cf-objective-ref="{obj_ref}"` when `q.get("objective_ref")` is truthy.
- **L492–513 `_render_activities`** — emits `<div class="activity-card"...>` with `data-cf-objective-ref="{obj_ref}"` similarly.

Both are HTML-escaped via `html_mod.escape`. Refs flow from curriculum JSON (e.g., `CO-01`, `TO-05`) into the page DOM. The attribute can carry course-level OR week-scoped IDs (`W03-CO-01`) — ingest must not assume a particular form.

### 1.2 Consume-side anchor (Worker K's pattern)

`Trainforge/parsers/html_content_parser.py`:
- **L317–321 `_extract_sections`** — Worker K's `data-cf-teaching-role` regex scan. Operates on `section_html` (HTML between heading `i` and heading `i+1`).
- Pattern: `re.findall(r'data-cf-teaching-role="([^"]*)"', section_html)`, dedup to `distinct_roles`, then expose on `ContentSection.teaching_role` (single) and `ContentSection.teaching_roles` (list).

**JSL-03 mirror:** add a second `re.findall(r'data-cf-objective-ref="([^"]*)"', section_html)` right next to it; dedup. Surface on `ContentSection.objective_refs: List[str]` (plural — multiple activities per section may cite different LOs; we want ALL of them on the chunk).

Naming rationale: single-valued `teaching_role` makes sense because Worker K collapsed "exactly one distinct role in the section" into a scalar. For objective refs, multi-value is the common case (an activity targets CO-01, a self-check targets CO-02, both live in the same section) — a list-only field keeps semantics honest.

### 1.3 Page-level fallback

When the page has no sections (rare; happens when `_chunk_content` falls into the `if not item["sections"]:` branch at L1015), the section-level `objective_refs` never attach. For that path we additionally scan the raw HTML once at parse time and expose a page-level list on `ParsedHTMLModule`.

**Parser output additions:**
- `ContentSection.objective_refs: List[str]` (default `[]`)
- `ParsedHTMLModule.objective_refs: List[str]` (default `[]`) — union of all section-level values + any `data-cf-objective-ref` on the page that fell outside sections.

### 1.4 Attachment in `process_course.py`

Currently `_create_chunk` (L1223–1277) calls `self._extract_objective_refs(item)` (L1676) which returns a flat list from JSON-LD / parsed LO objects. That call has no `section_heading` context.

**Change:** pass `section_heading` through to `_extract_objective_refs`, then merge in:
- section-scoped `objective_refs` when a matching section is found (same heading-match logic as `_extract_section_metadata` at L1510)
- page-scoped `objective_refs` fallback when no section matches (chunks from the no-sections branch)

Merge semantics:
- Start with refs from `learning_objectives` (existing L1684–1692 logic; honors case policy per §2).
- Add section-level refs (normalized through the same case policy).
- Dedup order-preserving.

### 1.5 Item dict: `parsed_items` field

`_parse_html` (L903) builds the per-item dict. Append:
```python
"objective_refs": parsed.objective_refs,  # page-level
# sections already flow through via item["sections"]; each is a
# ContentSection that carries its own .objective_refs
```

### 1.6 Ordering of refs

Existing `learning_outcome_refs` are populated from `learning_objectives[].id` in iteration order. New activity/self-check refs append after, dedup-filtered. This preserves backward-compat ordering for pages without activity refs and appends new information for pages with them.

---

## 2. Case preservation design

### 2.1 Primary target

`Trainforge/process_course.py:1688`:
```python
normalized = obj_id.lower().strip()
```
becomes:
```python
preserve_case = os.getenv("TRAINFORGE_PRESERVE_LO_CASE", "").lower() == "true"
normalized = obj_id.strip() if preserve_case else obj_id.lower().strip()
```

`os` is already imported at module top (confirmed — L1860 uses `os.getenv`).

The `WEEK_PREFIX_RE.sub('', normalized)` at L1690 stays unchanged — week-prefix stripping is regex-based and orthogonal to case. BUT: `WEEK_PREFIX_RE` is defined as `re.compile(r'^w\d+-')` (confirmed at L712) which is case-sensitive lowercase-only. With case preservation on, a ref like `W03-CO-01` would bypass stripping and stay as `W03-CO-01`. That mirrors the emit-time casing, which is actually correct: the week-prefix logic exists to fold week-scoped IDs into their canonical CO-XX form, but it currently only matches lowercase. This is an existing latent defect documented in Wave 4's scope; the case-preservation flag exposes it but does not cause it. Document as a known behavior in the sub-plan's risk table below.

### 2.2 Grep audit (`learning_outcome|lo_ref|.lower()`)

Every `.lower()` call in `Trainforge/process_course.py`, classified:

| Line | Expression                                             | Category                                              | Action                               |
|------|--------------------------------------------------------|-------------------------------------------------------|--------------------------------------|
| 242  | `path_lower = path.lower()`                            | file-path check, not LO-ref                           | leave as-is                          |
| 300  | `bloom = obj.get("bloomLevel", "").lower()`            | Bloom level normalization, not LO-ref                 | leave as-is                          |
| 354  | `tag = raw.lower().strip()`                            | concept-tag normalization (shared with tag slugging)  | leave as-is                          |
| 443  | `verb = match.group(1).lower()`                        | Bloom verb                                            | leave as-is                          |
| 472  | `low = term.lower()`                                   | term lookup                                           | leave as-is                          |
| 478  | `if low in sentence.lower()`                           | sentence matching                                     | leave as-is                          |
| 524  | `if tag.lower() in _VOID_HTML_TAGS`                    | HTML parser                                           | leave as-is                          |
| 526  | `self._stack.append(tag.lower())`                      | HTML parser                                           | leave as-is                          |
| 532  | `tag = tag.lower()`                                    | HTML parser                                           | leave as-is                          |
| 562  | `key = statement.lower()`                              | objective statement dedup                             | leave as-is                          |
| 975  | `prefix = f"{self.course_code.lower()}_chunk_"`        | chunk id prefix                                       | leave as-is                          |
| **1688** | `normalized = obj_id.lower().strip()`              | **LO-ref normalization**                              | **FLIP (add env gate)**              |
| 1470 | `chunk_heading = ... .lower()`                         | heading match (case-insensitive by design)            | leave as-is                          |
| 1481 | `sec.get("heading", "").lower() == chunk_heading`      | heading match                                         | leave as-is                          |
| 1510 | `section.heading.lower() == chunk_heading`             | heading match                                         | leave as-is                          |
| 1569 | `h = heading.lower()`                                  | heading match                                         | leave as-is                          |
| 1656 | `text_lower = text.lower()`                            | concept-tag regex                                     | leave as-is                          |
| 1860 | `os.getenv(..., "").lower() == "true"`                 | env-var parse                                         | leave as-is                          |
| 2561 | `obj_id = (to.get("id") or "").lower()`                | build `valid_outcome_ids` for broken-refs detection   | **see §2.3 below**                   |
| 2566 | `ids.add(ws.lower())`                                  | week-scoped ID                                        | **see §2.3**                         |
| 2569 | `obj_id = (obj.get("id") or "").lower()`               | chapter-objective id                                  | **see §2.3**                         |
| 2574 | `ids.add(ws.lower())`                                  | week-scoped ID                                        | **see §2.3**                         |
| 2783 | `"id": to["id"].lower()`                               | build `course.json` learning_outcomes                 | **see §2.3**                         |
| 2792 | `"id": obj["id"].lower()`                              | build `course.json` learning_outcomes                 | **see §2.3**                         |

### 2.3 Downstream lowercasing sites (2561, 2569, 2566, 2574, 2783, 2792)

These build the canonical outcome-id set used for two things:
- **Integrity report** (`broken_refs`): chunks with `learning_outcome_refs` not in `valid_outcome_ids` get flagged.
- **`course.json` output**: final LibV2 artifact for the course.

If the chunk refs preserve case but `valid_outcome_ids` is lowercased, every preserved-case ref will read as "broken" in the integrity report. Similarly, `course.json` IDs will diverge from the chunk refs' casing.

**Decision:** keep the Worker M plan minimal — flip ONLY L1688 per the master plan's scope. For a user who flips the flag on:
- The integrity report's `broken_refs` will spike (cosmetic — chunks are emitted, files write, not a hard fail unless `--strict` is on).
- `course.json` `learning_outcomes[].id` remains lowercase; chunks are uppercase. Downstream consumers that join on ID must do case-folded comparison. **This is a documented Wave 4 migration item.**

`Trainforge/align_chunks.py` (Worker K's scope — DO NOT MODIFY) also lowercases at lines 161, 180, 208, 213, 217, 222, 252. When alignment runs (default on full-pipeline execution), chunk refs get RE-lowercased regardless of the flag state. This means the flag is only effective for the ingest-only code path (when alignment is skipped).

**User-facing documentation:** the PR body notes that users enabling `TRAINFORGE_PRESERVE_LO_CASE=true` should:
1. Skip the alignment stage (it re-lowercases).
2. Expect `broken_refs` to rise (until Wave 4 makes the valid-id set case-preserving too).
3. Keep `TRAINFORGE_VALIDATE_CHUNKS=false` if schema validation is strict about patterns — see §2.4.

### 2.4 chunk_v4.schema.json pattern risk

`schemas/knowledge/chunk_v4.schema.json` — `learning_outcome_refs.items` is:
```json
{ "type": "string" }
```
No pattern, no case constraint. Preserved-case refs validate cleanly. **No schema relaxation needed.** Users with `TRAINFORGE_VALIDATE_CHUNKS=true` are unaffected by the case flag.

### 2.5 Regression test design (extend `test_chunk_validation.py`)

Two new tests using `monkeypatch.setenv`:

```python
def test_preserve_case_flag_off_lowercases(monkeypatch):
    """Default env → refs lowercased (backward-compat)."""
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    # Build a minimal item with one LO id="TO-01", call
    # CourseProcessor._extract_objective_refs directly, assert ["to-01"].

def test_preserve_case_flag_on_preserves(monkeypatch):
    """Env set → refs preserve case."""
    monkeypatch.setenv("TRAINFORGE_PRESERVE_LO_CASE", "true")
    # Same item, expect ["TO-01"].
```

Use direct method call rather than full pipeline — simpler, deterministic. Import `CourseProcessor` and the `LearningObjective` dataclass; build a stub processor and invoke `_extract_objective_refs` with a stub item dict.

---

## 3. Packager stub inclusion design

### 3.1 Target file

`Courseforge/scripts/package_multifile_imscc.py:217–232`:

```python
manifest_xml = build_manifest(content_dir, course_code, course_title)

with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("imsmanifest.xml", manifest_xml)

    file_count = 0
    for week_dir in sorted(content_dir.glob("week_*")):
        if not week_dir.is_dir():
            continue
        for html_file in sorted(week_dir.glob("*.html")):
            zf.write(html_file, f"{week_dir.name}/{html_file.name}")
            file_count += 1

print(f"IMSCC created: {output_path}")
print(f"  Files: {file_count} HTML + 1 manifest = {file_count + 1} total")
```

### 3.2 Exact diff

Insert right after `zf.writestr("imsmanifest.xml", manifest_xml)` (L220), before the `file_count = 0` line (L222):

```python
# Include course_metadata.json (Worker J's Wave 2 classification stub) when present.
# Worker J emits this stub at content-dir root next to the IMSCC; bundling it
# inside the zip makes the IMSCC self-contained for downstream consumers
# (Trainforge already supports both zip-root and sibling paths).
stub_path = content_dir / "course_metadata.json"
stub_included = False
if stub_path.exists():
    zf.write(stub_path, stub_path.name)
    stub_included = True
```

Then adjust the final print. The current line is:
```python
print(f"  Files: {file_count} HTML + 1 manifest = {file_count + 1} total")
```
Change to include the stub only when present:
```python
if stub_included:
    total = file_count + 2  # HTML files + manifest + stub
    print(f"  Files: {file_count} HTML + 1 manifest + 1 course_metadata.json = {total} total")
else:
    total = file_count + 1
    print(f"  Files: {file_count} HTML + 1 manifest = {total} total")
```

### 3.3 Regression test design (extend `test_packager_default.py`)

Two new tests. Reuse the existing `content_dir_with_courseJson` / `content_dir_no_courseJson` fixtures and add a third for "has both stub and valid pages":

```python
def test_packager_includes_course_metadata_when_present(
    self, content_dir_with_courseJson, tmp_path, capsys,
):
    """Content dir with course_metadata.json → stub bundled at zip root."""
    # Add valid pages so validation passes.
    (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
        _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
    )
    (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
        _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
    )
    # Add the stub (any JSON is fine — packager doesn't parse it).
    (content_dir_with_courseJson / "course_metadata.json").write_text(
        json.dumps({"courseCode": "TEST_101", "taxonomy": "stub"}), encoding="utf-8",
    )
    output = tmp_path / "out.imscc"
    package_imscc(
        content_dir_with_courseJson, output, "TEST_101", "Test Course",
    )
    assert output.exists()
    # Open the zip and verify the stub is at root.
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "course_metadata.json" in names, (
        f"expected course_metadata.json at zip root; got {names}"
    )

def test_packager_skips_stub_when_absent(
    self, content_dir_with_courseJson, tmp_path, capsys,
):
    """Content dir without course_metadata.json → zip has manifest + html only."""
    (content_dir_with_courseJson / "week_01" / "week_01_overview.html").write_text(
        _page_html(["TO-01", "CO-01", "CO-02"]), encoding="utf-8",
    )
    (content_dir_with_courseJson / "week_03" / "week_03_overview.html").write_text(
        _page_html(["TO-01", "CO-03", "CO-04"]), encoding="utf-8",
    )
    output = tmp_path / "out.imscc"
    package_imscc(
        content_dir_with_courseJson, output, "TEST_101", "Test Course",
    )
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "imsmanifest.xml" in names
    assert "course_metadata.json" not in names
```

Note: `content_dir_with_courseJson` puts `course.json` at the content-dir root for validator auto-discovery. This is a DIFFERENT file from `course_metadata.json`. The fixture is compatible as-is.

### 3.4 Import update

The test imports `zipfile` at top; confirm already present in the test file. Current `test_packager_default.py` imports: `json`, `sys`, `pathlib`, `pytest`. Add `import zipfile`.

---

## 4. JSL-03 regression test design (new file `test_activity_objective_ref.py`)

Three tests:

```python
def test_activity_objective_ref_parses_into_learning_outcome_refs(monkeypatch, tmp_path):
    """Activity with data-cf-objective-ref → chunk's learning_outcome_refs contains it."""
    # Default env → refs lowercased.
    monkeypatch.delenv("TRAINFORGE_PRESERVE_LO_CASE", raising=False)
    # Synthesize a page with <h2>Section</h2> + <div class="activity-card" data-cf-objective-ref="CO-05">
    # Build an IMSCC-like item structure, process with _chunk_content, assert chunk has "co-05".

def test_self_check_objective_ref_parses_into_learning_outcome_refs(monkeypatch, tmp_path):
    """Self-check with data-cf-objective-ref → chunk's learning_outcome_refs contains it."""
    # Same shape, .self-check class.

def test_activity_objective_ref_deduped(monkeypatch, tmp_path):
    """Same ref on multiple activities in one section → single entry, no dup."""
    # Two activity-cards with data-cf-objective-ref="CO-05", one chunk, assert refs == ["co-05"].
```

**Isolation strategy:** use `CourseProcessor` in-memory. Build a fake `parsed_items` list with the synthesized HTML and call `_chunk_content` directly. Avoid the full IMSCC-on-disk path — deterministic, fast.

**Minimum processor shim:** `CourseProcessor.__init__` takes `imscc_path`, `course_code`, `output_dir`, and optionally `capture`. We can instantiate against a dummy path since we'll only call `_chunk_content`.

Alternative (simpler): call `HTMLContentParser.parse(html)` directly to verify the parser extension; separately call `_extract_objective_refs(item)` to verify the process_course merge logic. Two-layer tests, each fast.

**Chosen approach:** two-layer tests.
- Layer 1 (parser): `HTMLContentParser.parse(html)` → assert `parsed.sections[0].objective_refs == ["CO-05"]` and `parsed.objective_refs == ["CO-05"]`.
- Layer 2 (process_course): build a minimal `item` dict with the parsed module's fields + one section, call `CourseProcessor._extract_objective_refs(item, section_heading=...)` → assert the ref is present (case per env var).

---

## 5. Exact file change summary

### 5.1 Modified

**`Trainforge/parsers/html_content_parser.py`:**
- Add `objective_refs: List[str]` default `[]` to `ContentSection` dataclass (after `teaching_roles`).
- Add `objective_refs: List[str]` default `[]` to `ParsedHTMLModule` dataclass (after `suggested_assessment_types`).
- In `_extract_sections` (after L321 teaching_role scan): add `re.findall(r'data-cf-objective-ref="([^"]*)"', section_html)`, dedup+sort, surface on the `ContentSection`.
- In `parse` method (after section extraction): compute page-level `objective_refs` as sorted union of all section-level values AND any `data-cf-objective-ref` in the full html.
- Pass `objective_refs` into `ParsedHTMLModule` constructor.

**`Trainforge/process_course.py`:**
- L1688: gate on `TRAINFORGE_PRESERVE_LO_CASE`.
- L946-ish in `_parse_html`: add `"objective_refs": parsed.objective_refs` to the item dict.
- `_extract_objective_refs`: accept optional `section_heading` parameter. After current LO-id extraction, also merge in:
  - section-scoped `objective_refs` from matching `item["sections"]` (heading-match via same logic as L1510)
  - page-scoped fallback from `item["objective_refs"]` when no section matches
- `_create_chunk` call site (L1273): pass `section_heading` through.
- Respect the case policy when normalizing the merged refs (apply `lower()` or not per env).

**`Courseforge/scripts/package_multifile_imscc.py`:**
- L220-228: add stub inclusion block + update the summary print line.

### 5.2 Test files

**Modified:**
- `Trainforge/tests/test_chunk_validation.py` — +2 case-preservation tests.
- `Courseforge/scripts/tests/test_packager_default.py` — +2 stub-inclusion tests + `import zipfile`.

**New:**
- `Trainforge/tests/test_activity_objective_ref.py` — 3 JSL-03 tests.
- `plans/kg-quality-review-2026-04/worker-m-subplan.md` — this file.

### 5.3 Untouched (per master plan constraints)

- `Trainforge/align_chunks.py` — Worker K's Wave 2 scope.
- `schemas/knowledge/chunk_v4.schema.json` — pattern is already empty; no change needed.
- `Trainforge/process_course.py` lines 2561/2569/2566/2574/2783/2792 — documented but left for Wave 4 migration.

---

## 6. Risk table

| Risk                                                                 | Mitigation                                                                                  |
|----------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Case-preservation flag + align_chunks runs → refs get re-lowercased  | Documented in PR body; flag only effective for ingest-only runs.                            |
| `valid_outcome_ids` lowercasing makes preserved-case refs "broken"   | Documented — cosmetic impact on integrity report; hard-fail only under `--strict`.          |
| `WEEK_PREFIX_RE` is lowercase-only; preserved-case `W03-CO-01` bypasses strip | Documented — pre-existing defect, flag surfaces but doesn't cause. Wave 4 scope.     |
| Activity-ref attached to wrong section due to heading drift          | Falls back to page-level refs when no section match — safe default.                          |
| Stub included in zip but Trainforge looks for sibling file first     | Non-issue — Trainforge checks zip-root first per Worker J's Wave 2 consume update.          |
| `course_metadata.json` path collision with `imsmanifest.xml`         | Different filenames; no collision possible.                                                  |

---

## 7. Verification sequence

```bash
python3 -m ci.integrity_check
source venv/bin/activate
pytest Trainforge/tests/test_activity_objective_ref.py \
       Trainforge/tests/test_chunk_validation.py \
       Courseforge/scripts/tests/test_packager_default.py -x

# Full regression suite
pytest lib/tests/ Trainforge/tests/ Courseforge/scripts/tests/ MCP/tests/ -q
```

Expected: ≥675 passing (Wave 2 baseline 674 + 5 new; 2 case-preservation + 3 JSL-03; the 2 packager stub tests may overlap existing test class count).

---

## 8. Out of scope (per master plan)

- Do NOT modify `Trainforge/align_chunks.py` (Worker K's Wave 2 scope).
- Do NOT touch `schemas/knowledge/chunk_v4.schema.json`.
- Do NOT migrate existing LibV2 chunks.
- Do NOT flip `TRAINFORGE_PRESERVE_LO_CASE` default to true.
- Do NOT change Courseforge emit (separate from packager stub cleanup).
- Main branch untouched; PR targets `dev-v0.2.0`.
