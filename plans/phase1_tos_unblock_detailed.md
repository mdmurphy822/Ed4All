# Phase 1 Detailed Execution Plan — Courseforge ToS Unblock

Refines `/home/user/Ed4All/plans/phase1_tos_unblock.md` into atomic, individually-verifiable subtasks. Each subtask has a unique, deterministic verification command. The execution worker should NOT need to re-explore the codebase.

---

## Investigation findings (locked)

- **`Courseforge/scripts/generate_course.py::generate_week` is purely deterministic.** It consumes a fully-built `week_data` dict (lines 1761-2081) and emits HTML. Section prose comes from `week_data["content_modules"][i]["sections"][j]["paragraphs"]` (rendered at `_render_content_sections:990-1104`, line 1059). `_render_objectives` (`:833-873`), `_render_self_check` (`:898-951`), `_render_activities` (`:1107-1148`), `_render_flip_cards` (`:876-895`) are all template fillers — no LLM hooks today.
- **The `week_data` payload is built upstream in `MCP/tools/_content_gen_helpers.py::build_week_data` (`:1576-1746`).** Its prose source is DART-parsed paragraphs (`primary_topic["paragraphs"]`, lines 1617-1620). When DART staging is empty, prose lists are empty — no LLM is called in the default pipeline.
- **The LLM-authored surface only opens under Wave 74 dispatch.** `MCP/core/executor.py:833-868` short-circuits to `dispatcher.dispatch_task("content-generator", ...)` when (a) `ED4ALL_AGENT_DISPATCH=true`, (b) a dispatcher is wired, (c) `agent_type=="content-generator"` is in `AGENT_SUBAGENT_SET` at line 228. The subagent's own Claude Code session is the LLM (`Courseforge/agents/content-generator.md`, 742 lines of authoring directives — Pattern 22 prevention, OSCQR, scenario domains, JSON-LD source references).
- **Decision capture in `_generate_course_content`** uses `phase="content-generator"` (canonical, `decision_event.schema.json:53`) and emits one `content_structure` event per run (pipeline_tools.py:2774-2802). Per-LLM-call attribution is absent. Phase 1 adds `content_generator_call`.
- **Reusable provider machinery is in `Trainforge/generators/`:** `_curriculum_provider.py:158-263` is the canonical line-for-line template; `_openai_compatible_client.py` carries the JSON / retry / decision-capture HTTP composition; `_anthropic_provider.py`, `_together_provider.py`, `_local_provider.py` carry env-var constants reusable as imports.
- **Test directory is `Courseforge/tests/`** (registered in `pytest.ini:11`). It already hosts `__init__.py`, `test_chunk_template_consumability.py`, `test_content_generator_spec_misconceptions.py` — same import-path pattern as the curriculum-alignment test (`PROJECT_ROOT = Path(__file__).resolve().parents[2]`).
- **The decision-event enum** at `decision_event.schema.json:63-136` is alphabetised. `content_generator_call` slots between `content_structure` (line 83) and `curriculum_alignment_call` (line 84).
- **`docs/LICENSING.md`** is 128 lines. The line-28 statement that needs correction reads literally `"Why this is fine for Ed4All: Claude Code does not generate training data on this project. Training-data synthesis routes through ..."`. The "Synthesis providers" table is at lines 46-60. The "Curriculum alignment shares the synthesis provider stack" anchor is lines 83-87.
- **Root `CLAUDE.md`** Opt-In Behavior Flags table at `:702-729`. The `CURRICULUM_ALIGNMENT_PROVIDER` row is line 729 (the last row). New `COURSEFORGE_PROVIDER` row inserts BEFORE it (alphabetic order: `COURSEFORGE_PROVIDER` < `CURRICULUM_ALIGNMENT_PROVIDER`).
- **`Courseforge/CLAUDE.md`** is 406 lines. The "Quick Start" block is near the top (lines ~10-20). The provider-selection note inserts after "Quick Start" and before "Workflow Pipelines."
- **The Wave 74 short-circuit guard** at `executor.py:833-839` is the precise integration point. The new guard inserts at line 833 (BEFORE the existing `if (_agent_dispatch_enabled() and self.dispatcher is not None ...)` block) so that `COURSEFORGE_PROVIDER` set + `agent_type=="content-generator"` falls through to the in-process `_generate_course_content`. Other Wave-74 agents (course-outliner, oscqr-course-evaluator, etc.) keep dispatching unchanged.

## Pre-resolved decisions

1. **`COURSEFORGE_PROVIDER` default.** `anthropic` (mirrors `CURRICULUM_ALIGNMENT_PROVIDER`'s `DEFAULT_PROVIDER` at `_curriculum_provider.py:89`). Rationale: flipping default to `local` silently breaks any operator running content-generator under `ED4ALL_AGENT_DISPATCH=true` who hasn't installed Ollama. The recommendation to use `local` for ToS-clean output lands in `docs/LICENSING.md` and `CLAUDE.md`, not in code.
2. **Where the LLM call lives.** **Today, only inside the Wave-74 subagent dispatch path** — there is NO Python-side LLM in the default `_generate_course_content` flow. Phase 1 introduces a new in-process LLM seam as a sibling to (not a replacement of) the subagent path. The seam is `MCP/tools/_content_gen_helpers.py::build_week_data` (NOT `generate_course.py::generate_week`). Specifically: when `content_provider` is non-None and `week_topics` is empty (or operator opted in via env var), `build_week_data` calls `content_provider.generate_page(...)` to produce the prose paragraphs that get embedded in the `content_modules[*].sections[*].paragraphs` list. The deterministic `generate_week` path then renders those paragraphs unchanged. This keeps the surgical edit out of the 2296-line `generate_course.py`.
3. **`generate_page` return type.** `str` for Phase 1. Comment in the docstring explicitly notes `# Phase 2: returns Block` so the version contract is unambiguous. The env var name does NOT change in Phase 2 — only the return type widens.
4. **`--courseforge-provider` CLI flag.** No, env-var only for Phase 1. Mirrors `CURRICULUM_ALIGNMENT_PROVIDER`'s Wave-137 follow-up history. CLI flag can be a Phase-1.5 addition if operators want symmetry.
5. **Wave-74 short-circuit semantics.** Any non-empty value of `COURSEFORGE_PROVIDER` short-circuits the subagent dispatch when `agent_type == "content-generator"`. Simpler mental model: "operator wants the provider, not the subagent." `=local`, `=together`, `=anthropic` all force the in-process path. `unset` keeps Wave-74 behaviour byte-identical.

---

## Atomic subtasks

Estimated total LOC across all subtasks: ~700 (250 provider + 250 tests + 80 wire-in + 40 schema + 80 docs).

### A. Schema additions

#### Subtask 1: Add `content_generator_call` to decision_event enum
- **Files:** `/home/user/Ed4All/schemas/events/decision_event.schema.json:83-84`
- **Depends on:** none
- **Estimated LOC:** 1 line added
- **Change:** Insert `"content_generator_call",` between line 83 (`"content_structure",`) and line 84 (`"curriculum_alignment_call",`). Maintain trailing-comma + lexical-sort style.
- **Verification:** `python -c "import json; e=json.load(open('schemas/events/decision_event.schema.json'))['properties']['decision_type']['enum']; assert 'content_generator_call' in e; assert e.index('content_generator_call') == e.index('content_structure')+1"`

### B. Provider package skeleton

#### Subtask 2: Create `Courseforge/generators/__init__.py`
- **Files:** create `/home/user/Ed4All/Courseforge/generators/__init__.py`
- **Depends on:** none
- **Estimated LOC:** 1 (empty marker; optionally a `__all__` re-export)
- **Change:** Empty package marker file. No re-exports yet.
- **Verification:** `python -c "import importlib; importlib.import_module('Courseforge.generators')"` exits 0.

#### Subtask 3: Create `Courseforge/generators/_provider.py` with constants + class skeleton (no method bodies)
- **Files:** create `/home/user/Ed4All/Courseforge/generators/_provider.py` (~50 lines for this subtask)
- **Depends on:** Subtask 2
- **Estimated LOC:** 50
- **Change:** Module docstring (mirror `Trainforge/generators/_curriculum_provider.py:1-33`); imports of `SynthesisProviderError`, `ANTHROPIC_DEFAULT_MODEL`/`ANTHROPIC_ENV_API_KEY`, `LOCAL_DEFAULT_BASE_URL`/`LOCAL_DEFAULT_MODEL`/`LOCAL_ENV_API_KEY`/`LOCAL_ENV_BASE_URL`/`LOCAL_ENV_MODEL`, `OpenAICompatibleClient`, `TOGETHER_DEFAULT_BASE_URL`/`TOGETHER_DEFAULT_MODEL`/`TOGETHER_ENV_API_KEY`/`TOGETHER_ENV_MODEL` from the corresponding `Trainforge.generators._*_provider` modules. Constants: `ENV_PROVIDER="COURSEFORGE_PROVIDER"`, `DEFAULT_PROVIDER="anthropic"`, `SUPPORTED_PROVIDERS=("anthropic","together","local")`, `_DEFAULT_MAX_TOKENS=4096`, `_DEFAULT_TEMPERATURE=0.4`. Class `ContentGeneratorProvider` with `__init__` signature mirroring `_curriculum_provider.py:158-171` exactly, `pass` body. `__all__` list at bottom.
- **Verification:** `python -c "from Courseforge.generators._provider import ContentGeneratorProvider, ENV_PROVIDER, DEFAULT_PROVIDER, SUPPORTED_PROVIDERS; assert DEFAULT_PROVIDER=='anthropic'; assert SUPPORTED_PROVIDERS==('anthropic','together','local')"`

### C. Provider class branches

#### Subtask 4: Implement `ContentGeneratorProvider.__init__` `anthropic` branch
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py` (replace `pass` body)
- **Depends on:** Subtask 3
- **Estimated LOC:** ~30
- **Change:** Mirror `_curriculum_provider.py:172-210` line-for-line: resolve `provider`, validate against `SUPPORTED_PROVIDERS`, raise `ValueError` on miss, set `self._provider`/`_capture`/`_max_tokens`/`_temperature`. For `resolved_provider=="anthropic"`: resolve `self._model` from `model` arg / `ANTHROPIC_SYNTHESIS_MODEL` env / `ANTHROPIC_DEFAULT_MODEL`; resolve `api_key` from arg / `ANTHROPIC_ENV_API_KEY` env; raise `RuntimeError` when neither `anthropic_client` injected nor key present (preserve the test-injection escape hatch); store `_anthropic_client`, set `_oa_client=None`, `_base_url=None`. Together + local branches stub-raise `NotImplementedError` for now so this subtask is independently verifiable.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py::test_unknown_provider_raises_value_error Courseforge/tests/test_content_generator_provider.py::test_default_provider_is_anthropic_when_env_unset -v` PASSES (after Subtask 9). Standalone gate: `python -c "import os; os.environ['ANTHROPIC_API_KEY']='k'; from Courseforge.generators._provider import ContentGeneratorProvider; p=ContentGeneratorProvider(provider='anthropic', anthropic_client=object()); assert p._provider=='anthropic'"`.

#### Subtask 5: Implement `ContentGeneratorProvider.__init__` `together` branch
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py`
- **Depends on:** Subtask 4
- **Estimated LOC:** ~25
- **Change:** Replace the `together` `NotImplementedError` stub with the body mirroring `_curriculum_provider.py:211-236`: resolve model from `TOGETHER_ENV_MODEL` env, base URL from arg / `TOGETHER_DEFAULT_BASE_URL`, api_key from `TOGETHER_ENV_API_KEY` env, raise `RuntimeError` when neither `client` nor `api_key`. Construct `self._oa_client = OpenAICompatibleClient(base_url=..., model=..., api_key=..., capture=None, provider_label="together", client=client)`. Set `_anthropic_client=None`.
- **Verification:** `python -c "import os; os.environ['TOGETHER_API_KEY']='tk'; from Courseforge.generators._provider import ContentGeneratorProvider; p=ContentGeneratorProvider(provider='together'); assert p._provider=='together'; assert p._base_url.startswith('https://api.together.xyz')"`.

#### Subtask 6: Implement `ContentGeneratorProvider.__init__` `local` branch
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py`
- **Depends on:** Subtask 5
- **Estimated LOC:** ~22
- **Change:** Replace the `local` `NotImplementedError` stub. Mirror `_curriculum_provider.py:238-262`: model from `LOCAL_ENV_MODEL` env, api_key from `LOCAL_ENV_API_KEY` env defaulting to `"local"` (no RuntimeError when missing), base URL from `LOCAL_ENV_BASE_URL` env / `LOCAL_DEFAULT_BASE_URL`. Construct `self._oa_client = OpenAICompatibleClient(... provider_label="local" ...)`. Set `_anthropic_client=None`.
- **Verification:** `python -c "import os; os.environ.pop('LOCAL_SYNTHESIS_API_KEY', None); from Courseforge.generators._provider import ContentGeneratorProvider; p=ContentGeneratorProvider(provider='local'); assert p._provider=='local'; assert p._api_key=='local'"`.

### D. Provider `generate_page` + dispatch + decision capture

#### Subtask 7: Implement `_render_user_prompt`, `_dispatch_call`, `_call_anthropic` helpers
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py`
- **Depends on:** Subtask 6
- **Estimated LOC:** ~80
- **Change:** Add module-level constant `_SYSTEM_PROMPT` (~300-char condensation of `Courseforge/agents/content-generator.md` core directives: Pattern 22 prevention, color palette constraint, OSCQR alignment, source-grounding contract, "emit only the rendered HTML body — no preamble"). Add `_render_user_prompt(self, *, course_code, week_number, page_id, page_template, page_context) -> str` that JSON-serialises `page_context` and embeds the slotted `page_template` literal — no `concept_tags` or other non-courseforge fields. Add `_dispatch_call(self, user_prompt) -> tuple[str, int]` mirroring `_curriculum_provider.py:363-385`: anthropic → `_call_anthropic`, otherwise build messages with `_SYSTEM_PROMPT` + user_prompt, call `self._oa_client._post_with_retry(payload)` with `temperature`/`max_tokens` from instance, extract via `_oa_client._extract_text(body)`. Add `_call_anthropic(self, user_prompt) -> tuple[str, int]` mirroring `_curriculum_provider.py:387-434` exactly, including the lazy-import-on-first-call pattern.
- **Verification:** `python -c "from Courseforge.generators._provider import ContentGeneratorProvider; assert hasattr(ContentGeneratorProvider, '_dispatch_call') and hasattr(ContentGeneratorProvider, '_call_anthropic') and hasattr(ContentGeneratorProvider, '_render_user_prompt')"`.

#### Subtask 8: Implement `generate_page` public method + decision-capture emit
- **Files:** `/home/user/Ed4All/Courseforge/generators/_provider.py`
- **Depends on:** Subtask 7
- **Estimated LOC:** ~60
- **Change:** Public method `generate_page(self, *, course_code: str, week_number: int, page_id: str, page_template: str, page_context: Dict[str, Any]) -> str`. Docstring explicitly says: `"""Returns rendered HTML as str (Phase 1). Phase 2: will return Block."""`. Validate inputs: empty `page_id` raises `ValueError("page_id required")`; empty `course_code` raises `ValueError("course_code required")`. Build user prompt via `_render_user_prompt`, call `self._dispatch_call(user_prompt)` returning `(text, retry_count)`. Skip role-validation (no enum constraint at this surface — caller validates HTML). Call `self._emit_decision(course_code=..., week_number=..., page_id=..., retry_count=..., raw_text=text)`. Return `text`. Add private `_emit_decision` mirroring `_curriculum_provider.py:477-516`: `decision_type="content_generator_call"`, decision string interpolating `page_id`/`course_code`/`week_number`/`provider`/`model`/`retry_count`, rationale ≥20 chars interpolating same plus optional `base_url` plus `len(raw_text)` chars plus the operator-control + ToS-routing justification. Wrap in `try/except` so capture failure never breaks the caller (mirrors curriculum-alignment).
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -k "decision_capture" -v` PASSES (after Subtask 13).

### E. Test infrastructure

#### Subtask 9: Create test scaffolding `Courseforge/tests/test_content_generator_provider.py`
- **Files:** create `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtask 3 (provider class importable)
- **Estimated LOC:** ~80 (helpers only; tests added in subsequent subtasks)
- **Change:** Mirror `Trainforge/tests/test_curriculum_alignment_provider.py:1-100`: module docstring, sys.path insertion (`PROJECT_ROOT = Path(__file__).resolve().parents[2]`), imports of `ContentGeneratorProvider`/`DEFAULT_PROVIDER`/`ENV_PROVIDER`/`SUPPORTED_PROVIDERS`/`SynthesisProviderError`. Helpers: `_success_body(content, *, model="test-model")` returning the OpenAI-compatible JSON envelope; `_make_client(handler)` returning `httpx.Client(transport=httpx.MockTransport(handler))`; `_FakeCapture` class (events list + `log_decision` appender). Helper to build a sample `page_context` dict: `_sample_page_context()` with `{"objectives": [...], "key_terms": [...], "section_headings": [...], "primary_topic": {...}}`.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py --collect-only` reports collection (no tests yet, no errors).

### F. Per-branch tests

#### Subtask 10: Add construction tests
- **Files:** `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtasks 4-6, 9
- **Estimated LOC:** ~40
- **Change:** Add tests mirroring curriculum-alignment counterpart:
  - `test_unknown_provider_raises_value_error` (`pytest.raises(ValueError)` on `provider="bogus"`).
  - `test_default_provider_is_anthropic_when_env_unset(monkeypatch)` — `delenv(ENV_PROVIDER)`, `setenv("ANTHROPIC_API_KEY","k")`, construct with `anthropic_client=object()`, assert `_provider=="anthropic"`, assert `DEFAULT_PROVIDER=="anthropic"`.
  - `test_env_var_selects_provider(monkeypatch)` — `setenv(ENV_PROVIDER,"local")`, construct with default args, assert `_provider=="local"`.
  - `test_supported_providers_set_is_three`.
  - `test_anthropic_backend_missing_api_key_raises(monkeypatch)`.
  - `test_together_backend_missing_api_key_raises(monkeypatch)`.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -k "construction or unknown or default or env_var or supported or missing_api_key" -v` PASSES (6 tests).

#### Subtask 11: Add `local` happy-path test
- **Files:** `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtasks 6, 8, 9
- **Estimated LOC:** ~25
- **Change:** `test_local_backend_routes_to_local_base_url(monkeypatch)`: `delenv("LOCAL_SYNTHESIS_API_KEY")`, `setenv("LOCAL_SYNTHESIS_BASE_URL","http://localhost:11434/v1")`, mock handler returns `_success_body("<section><h2>Topic</h2><p>"+"alpha "*100+"</p></section>")`. Construct `ContentGeneratorProvider(provider="local", client=_make_client(handler))`. Call `generate_page(course_code="DEMO_101", week_number=1, page_id="week_01_content_01_intro", page_template="<!--TEMPLATE-->", page_context=_sample_page_context())`. Assert returned string contains `"<section>"`. Assert handler called once. Assert request URL is `"http://localhost:11434/v1/chat/completions"`.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py::test_local_backend_routes_to_local_base_url -v` PASSES.

#### Subtask 12: Add `together` and `anthropic` happy-path tests
- **Files:** `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtasks 5, 7, 9
- **Estimated LOC:** ~50
- **Change:** `test_together_backend_returns_html(monkeypatch)`: `setenv("TOGETHER_API_KEY","tk")`, mock handler returns `_success_body("<p>Body</p>")`. Construct `provider="together"`. Call `generate_page`. Assert response endpoint contains `api.together.xyz/v1/chat/completions`. `test_anthropic_backend_returns_html(monkeypatch)`: `setenv("ANTHROPIC_API_KEY","ak")`, fake `_FakeMessages.create` returning `{"content":[{"type":"text","text":"<p>Body</p>"}]}`, fake `_FakeClient` with `messages = _FakeMessages()`. Construct with `anthropic_client=_FakeClient()`. Call `generate_page`. Assert returned text contains `"<p>"`.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -k "together_backend_returns_html or anthropic_backend_returns_html" -v` PASSES.

#### Subtask 13: Add decision-capture tests
- **Files:** `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtask 8, 11
- **Estimated LOC:** ~35
- **Change:** `test_decision_capture_fires_with_page_id_and_provider_in_rationale(monkeypatch)` — set up local provider with `_FakeCapture`, call `generate_page(course_code="DEMO_101", week_number=3, page_id="week_03_content_01_topic", ...)`. Assert exactly one event with `decision_type=="content_generator_call"`, `len(rationale)>=20`, rationale contains `"course_code=DEMO_101"` (or just `"DEMO_101"`), `"week_number=3"` (or `"week 3"`), `"week_03_content_01_topic"`, `"provider=local"`, `"model="`. Assert `decision` field contains `"week_03_content_01_topic"`. `test_empty_page_id_raises_value_error` — assert `ValueError` on empty `page_id`. `test_empty_course_code_raises_value_error` — analogous.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py -k "decision_capture or empty_page_id or empty_course_code" -v` PASSES (3 tests).

### G. Pipeline integration test

#### Subtask 14: Add pipeline-level integration test (env var → provider call observed)
- **Files:** `/home/user/Ed4All/Courseforge/tests/test_content_generator_provider.py`
- **Depends on:** Subtasks 16-18 (pipeline_tools wire-in)
- **Estimated LOC:** ~50
- **Change:** `test_pipeline_tools_routes_through_provider_when_env_set(monkeypatch, tmp_path)` — set `COURSEFORGE_PROVIDER=local`, `LOCAL_SYNTHESIS_BASE_URL` pointing at a `MockTransport` URL the test owns, fabricate a minimal `Courseforge/exports/<project>/project_config.json` + empty staged-DART dir under `tmp_path`. Drive `_generate_course_content` (import via `MCP.tools.pipeline_tools`). Assert handler observed at least one POST to `/v1/chat/completions`. Assert NO Anthropic SDK import in `sys.modules` post-call (or alternatively monkeypatch `anthropic` to a sentinel and assert sentinel never read). Skip with `pytest.skip` when `httpx` not importable.
- **Verification:** `pytest Courseforge/tests/test_content_generator_provider.py::test_pipeline_tools_routes_through_provider_when_env_set -v` PASSES.

### H. Wire-in `pipeline_tools.py`

#### Subtask 15: Read `COURSEFORGE_PROVIDER` env var in `_generate_course_content` and instantiate provider
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:2774-2802` (immediately before the existing `DecisionCapture` block)
- **Depends on:** Subtask 8
- **Estimated LOC:** ~15
- **Change:** Insert just before the `try: from lib.decision_capture import DecisionCapture` block: `_courseforge_provider_env = os.environ.get("COURSEFORGE_PROVIDER", "").strip()`. After `capture` is constructed: `content_provider = None; if _courseforge_provider_env: from Courseforge.generators._provider import ContentGeneratorProvider; content_provider = ContentGeneratorProvider(capture=capture)`. Wrap the import + construction in `try/except` so a missing prereq (e.g. anthropic key) raises a clear error EARLY rather than silently falling back to the deterministic path.
- **Verification:** `grep -n "COURSEFORGE_PROVIDER" MCP/tools/pipeline_tools.py` shows exactly one match at line ~2774. `pytest Courseforge/tests/test_content_generator_provider.py::test_pipeline_tools_routes_through_provider_when_env_set -v` PASSES (after Subtask 16-17).

#### Subtask 16: Thread `content_provider` into `build_week_data`
- **Files:** `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py:1576` (signature) and `1747-1750` (`_build_content_modules_dynamic` signature)
- **Depends on:** Subtask 15
- **Estimated LOC:** ~10
- **Change:** Add `content_provider: Optional["ContentGeneratorProvider"] = None` kwarg to `build_week_data`. Pass through to `_build_content_modules_dynamic(content_provider=content_provider, course_code=course_code, week_num=week_num, ...)` (lines 1647-1651).
- **Verification:** `python -c "import inspect; from MCP.tools._content_gen_helpers import build_week_data; assert 'content_provider' in inspect.signature(build_week_data).parameters"`.

#### Subtask 17: Implement provider call inside `_build_content_modules_dynamic`
- **Files:** `/home/user/Ed4All/MCP/tools/_content_gen_helpers.py` near `_build_content_modules_dynamic` (around line 1748-1810)
- **Depends on:** Subtask 16
- **Estimated LOC:** ~30
- **Change:** When `content_provider is not None`, before falling back to DART-paragraph synthesis, call `content_provider.generate_page(course_code=course_code, week_number=week_num, page_id=f"week_{week_num:02d}_content_{i:02d}", page_template="<!--CONTENT_MODULE-->", page_context={"objectives": [...], "topic_heading": ..., "key_terms": [...]})`. Parse the returned HTML into `(heading, paragraphs[])` via a minimal regex (not a full HTML parser — Phase 2 makes this a `Block`). Emit those into the `sections[*]["paragraphs"]` list. When the provider returns empty / parse fails, fall back to legacy DART-paragraph path WITHOUT raising (warn log). Existing legacy path is unchanged when `content_provider is None`.
- **Verification:** `pytest Courseforge/scripts/tests/test_generate_course_sourcerefs.py -v` PASSES (regression — provider=None path unchanged). `pytest Courseforge/tests/test_content_generator_provider.py::test_pipeline_tools_routes_through_provider_when_env_set -v` PASSES.

#### Subtask 18: Pass `content_provider` from `_generate_course_content` into `build_week_data`
- **Files:** `/home/user/Ed4All/MCP/tools/pipeline_tools.py:2858-2865` (the `build_week_data` call)
- **Depends on:** Subtasks 15, 16
- **Estimated LOC:** 1
- **Change:** Add `content_provider=content_provider` kwarg to the `_cgh.build_week_data(...)` call site.
- **Verification:** `grep -A1 "_cgh.build_week_data" MCP/tools/pipeline_tools.py | grep content_provider` returns the line.

### I. Wave-74 short-circuit guard

#### Subtask 19: Add `COURSEFORGE_PROVIDER` short-circuit guard in `_invoke_tool`
- **Files:** `/home/user/Ed4All/MCP/core/executor.py:833-839` (immediately before the `if (_agent_dispatch_enabled() and ...` block)
- **Depends on:** none (independent of provider class)
- **Estimated LOC:** ~10
- **Change:** Insert new guard expression so the existing `if` block becomes:
  ```python
  _courseforge_provider_set = bool(os.environ.get("COURSEFORGE_PROVIDER", "").strip())
  _force_inprocess_for_courseforge = (
      _courseforge_provider_set and agent_type == "content-generator"
  )
  if (
      _agent_dispatch_enabled()
      and self.dispatcher is not None
      and isinstance(agent_type, str)
      and agent_type in AGENT_SUBAGENT_SET
      and hasattr(self.dispatcher, "dispatch_task")
      and not _force_inprocess_for_courseforge
  ):
  ```
  Add a `logger.info` line inside the new fall-through case so operators see "COURSEFORGE_PROVIDER set; bypassing content-generator subagent dispatch."
- **Verification:** `grep -n "_force_inprocess_for_courseforge" MCP/core/executor.py` returns exactly two matches (definition + use in the `if`).

### J. Documentation

#### Subtask 20: Correct the false statement at `docs/LICENSING.md:28`
- **Files:** `/home/user/Ed4All/docs/LICENSING.md:28`
- **Depends on:** none
- **Estimated LOC:** ~6 (replace one line with multi-line corrected text)
- **Change:** Replace the sentence `"Why this is fine for Ed4All: Claude Code does not generate training data on this project. Training-data synthesis routes through the dedicated providers in Trainforge/generators/ (see next section). Code Claude writes for the orchestrator does not become a training example."` with: `"Why this is fine for Ed4All — with one caveat: Claude Code does not generate training data on this project, EXCEPT through the Courseforge content-generator subagent under ED4ALL_AGENT_DISPATCH=true when COURSEFORGE_PROVIDER is unset. In that configuration, the subagent's Claude Code session authors HTML prose that Trainforge ingests as training chunks — i.e. it touches training data. Setting COURSEFORGE_PROVIDER=local (or together) routes the same surface through a license-clean provider. See the Synthesis providers table below."`
- **Verification:** `grep -A1 "Why this is fine" docs/LICENSING.md | head -2` shows the new text.

#### Subtask 21: Add `COURSEFORGE_PROVIDER=*` rows to `docs/LICENSING.md` providers table
- **Files:** `/home/user/Ed4All/docs/LICENSING.md:46-60` (synthesis providers table)
- **Depends on:** Subtask 20
- **Estimated LOC:** ~3 rows
- **Change:** Add three rows immediately after `claude_session` (line 49) and before `together (Llama)` (line 50): one each for `COURSEFORGE_PROVIDER=anthropic` (Anthropic Commercial / `No`), `COURSEFORGE_PROVIDER=together` (Llama 3.3 / Together AI ToS / `Yes`), `COURSEFORGE_PROVIDER=local` (Apache 2.0 / `Yes` / **Recommended for ToS-clean Courseforge content**). Use the same `|`-separated table syntax.
- **Verification:** `grep "COURSEFORGE_PROVIDER" docs/LICENSING.md | wc -l` returns at least 3.

#### Subtask 22: Add "Courseforge content-generator shares the synthesis provider stack" section
- **Files:** `/home/user/Ed4All/docs/LICENSING.md:87` (insert after the "Curriculum alignment shares the synthesis provider stack" section)
- **Depends on:** Subtask 21
- **Estimated LOC:** ~6
- **Change:** New `### Courseforge content-generator shares the synthesis provider stack` heading. One paragraph: explains that Courseforge content-generator reuses `LOCAL_SYNTHESIS_*` / `TOGETHER_*` / `ANTHROPIC_SYNTHESIS_*` env vars; emits one `content_generator_call` decision event per page; recommends `local` for ToS-clean course generation. Note the Wave-74 short-circuit semantics: setting `COURSEFORGE_PROVIDER` overrides `ED4ALL_AGENT_DISPATCH=true` for the content-generator agent only.
- **Verification:** `grep -c "Courseforge content-generator shares" docs/LICENSING.md` returns 1.

#### Subtask 23: Add `COURSEFORGE_PROVIDER` row to root `CLAUDE.md` opt-in flags table
- **Files:** `/home/user/Ed4All/CLAUDE.md:728` (insert before the `CURRICULUM_ALIGNMENT_PROVIDER` row at line 729)
- **Depends on:** Subtask 22
- **Estimated LOC:** ~6 (single-row block, multi-line per the existing prose-row style)
- **Change:** Insert one row matching the prose-density of the `CURRICULUM_ALIGNMENT_PROVIDER` row at line 729. Content: name, values (`anthropic` / `together` / `local`), reuse-of-synthesis-env-vars note, default-deterministic-template-path note, Wave-74 short-circuit semantics, decision-event reference (`content_generator_call`), `docs/LICENSING.md` cross-link.
- **Verification:** `grep -B1 "CURRICULUM_ALIGNMENT_PROVIDER" CLAUDE.md | grep -c COURSEFORGE_PROVIDER` returns at least 1.

#### Subtask 24: Add provider-selection note to `Courseforge/CLAUDE.md`
- **Files:** `/home/user/Ed4All/Courseforge/CLAUDE.md` near the "Quick Start" block (lines ~10-20)
- **Depends on:** Subtask 23
- **Estimated LOC:** ~5
- **Change:** New `### Provider selection (Phase 1 ToS unblock)` subsection with one paragraph linking to root `CLAUDE.md` env-var table and `docs/LICENSING.md`. Do NOT duplicate the env-var table content. One sentence: "Set `COURSEFORGE_PROVIDER=local` to route content authoring through a license-clean local OSS provider; see root `CLAUDE.md` § Opt-In Behavior Flags for the env-var contract and `docs/LICENSING.md` for the ToS posture."
- **Verification:** `grep -n "Provider selection (Phase 1 ToS unblock)" Courseforge/CLAUDE.md` returns one match.

### K. End-to-end smoke

#### Subtask 25: Document the operator-facing smoke command in the plan + run it as a sanity check
- **Files:** none modified — operator/dev runs the command
- **Depends on:** all prior subtasks
- **Estimated LOC:** 0
- **Change:** Run the smoke sequence:
  ```bash
  export COURSEFORGE_PROVIDER=local
  export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
  export LOCAL_SYNTHESIS_MODEL=qwen2.5:14b-instruct-q4_K_M
  ed4all run textbook_to_course \
    --inputs Courseforge/inputs/textbooks/<book>/ \
    --course-code DEMO_101 \
    --weeks 4
  ```
- **Verification (3 commands):**
  1. `grep -l '"provider": "anthropic"' training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl 2>/dev/null` returns NO matches.
  2. `jq -r 'select(.decision_type=="content_generator_call") | .metadata.provider // "(unset)"' training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl | sort -u` returns only `local`.
  3. `ls Courseforge/exports/*/05_final_package/*.imscc` returns at least one IMSCC file.

---

## Execution sequencing

**Strict-serial (must run in order):**
- Subtask 1 (schema) — independent but unblocks Subtask 13 + 14.
- Subtasks 2 → 3 → 4 → 5 → 6 → 7 → 8 (provider class build-up; each depends on the prior).
- Subtask 9 (test scaffolding) → Subtasks 10-13 (per-branch tests) → Subtask 14 (integration test).
- Subtasks 15 → 16 → 17 → 18 (pipeline_tools.py wire-in chain). Subtask 14 depends on 15-18 collectively.

**Independently runnable in parallel (after their gating subtask):**
- Subtask 19 (executor.py guard) — can land any time after Subtask 1; no deps on the provider class.
- Subtasks 20-24 (documentation) — after Subtask 8 lands, all 5 doc subtasks can run in parallel; only Subtask 22 strictly needs Subtask 21 to land first (placement context).

**Suggested parallel batches:**
- Batch 1: Subtask 1, Subtask 19 (schema + executor guard, parallelisable, both isolated).
- Batch 2: Subtasks 2-8 serialised (provider class build).
- Batch 3: Subtasks 9-13 serialised (provider tests).
- Batch 4: Subtasks 15-18 serialised (pipeline wire-in).
- Batch 5: Subtask 14 (integration test — depends on 15-18).
- Batch 6: Subtasks 20-24 (docs — parallel after the code lands).
- Batch 7: Subtask 25 (operator smoke).

**Phase 2 dependency:** None. `generate_page` returns `str` for Phase 1; Phase 2 widens to `Block` without re-versioning the env var. Subtask 8's docstring explicitly carries the `# Phase 2: returns Block` comment to lock this in.

---

## Final smoke test

A single end-to-end verification an operator runs to prove Phase 1 landed:

```bash
# 1. Run the full unit + integration test suite for the new module:
pytest Courseforge/tests/test_content_generator_provider.py -v

# 2. Ensure no regression on the deterministic path (provider unset):
unset COURSEFORGE_PROVIDER
pytest Courseforge/scripts/tests/ -v

# 3. End-to-end with a local provider running:
export COURSEFORGE_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
export LOCAL_SYNTHESIS_MODEL=qwen2.5:14b-instruct-q4_K_M
ed4all run textbook_to_course \
  --inputs Courseforge/inputs/textbooks/<book>/ \
  --course-code DEMO_101 \
  --weeks 4

# 4. Verify zero Anthropic in the captures:
! grep -l '"provider": "anthropic"' \
    training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl

# 5. Verify provider attribution per page:
jq -r 'select(.decision_type=="content_generator_call") | .metadata.provider' \
    training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl \
    | sort -u | grep -qx local

# 6. Verify the IMSCC packaged successfully:
ls Courseforge/exports/*/05_final_package/*.imscc

# 7. Confirm the Wave-74 short-circuit fired (env-var override of dispatch):
ED4ALL_AGENT_DISPATCH=true COURSEFORGE_PROVIDER=local \
  ed4all run textbook_to_course --inputs ... --course-code DEMO_102 --weeks 1
# Then check the run logs for the message
# "COURSEFORGE_PROVIDER set; bypassing content-generator subagent dispatch."
```

Acceptance criteria: all four `pytest` invocations pass; commands 4-6 produce the documented exit codes; command 7 logs the bypass message at INFO level.
