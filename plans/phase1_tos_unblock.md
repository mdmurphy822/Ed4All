# Phase 1 — Courseforge ToS Unblock

**Goal:** Close Anthropic ToS exposure on Courseforge content generation by introducing
a `COURSEFORGE_PROVIDER` env-var-driven provider selection that mirrors the
existing `CURRICULUM_ALIGNMENT_PROVIDER` precedent (`Trainforge/generators/_curriculum_provider.py:88-263`).
After Phase 1, an operator sets `COURSEFORGE_PROVIDER=local` and runs `textbook_to_course`
end-to-end with zero Anthropic in the training-data path.

**Independent of:** Phase 2 (Block dataclass), Phase 3+ (two-pass rewrite).
This phase MUST work without any Phase 2 changes.

---

## 1. Investigation summary (locked findings)

### 1.1 Where the LLM call lives today

The Courseforge content-generation pipeline phase is `content_generation`
(see `config/workflows.yaml:818-838`). In `MCP/core/executor.py:147` the
`content-generator` agent maps to the in-process tool `generate_course_content`.
That tool is `MCP/tools/pipeline_tools.py::_generate_course_content`
(`pipeline_tools.py:2668-2982`); it delegates HTML rendering to
`Courseforge/scripts/generate_course.py::generate_week`. **This Python path
is fully deterministic — no LLM call.**

The LLM authoring surface only opens when **Wave 74 dispatch is enabled**
(`ED4ALL_AGENT_DISPATCH=true`, see `MCP/core/executor.py:225-261` and `:823-868`).
With the flag on, `TaskExecutor._invoke_tool` skips the in-process tool and
calls `dispatcher.dispatch_task(agent_type="content-generator", ...)`.
Under `LLM_MODE=local`, the dispatcher is `MCP/orchestrator/local_dispatcher.py::LocalDispatcher`,
whose `dispatch_task` (`local_dispatcher.py:227-308`) writes a pending agent task to the mailbox;
an outer Claude Code session reads `mailbox/pending/{task_id}.json`, dispatches
a subagent driven by `Courseforge/agents/content-generator.md`, and writes the
completion envelope. **The subagent's own Claude Code session IS the LLM** — there
is no Python-side `OpenAICompatibleClient` or `LLMBackend.from_spec()` call
on this path. That means the natural-language prose the templates leave as
slots (Pattern 22 theoretical-foundation paragraphs, real-world scenarios,
problem-solution walkthroughs, common-pitfall corrections, JSON-LD source
references, etc., per `content-generator.md:96-220`) is authored under
Anthropic Consumer Terms (Pro/Max session) — the most restrictive ToS row
in `docs/LICENSING.md:48-60`.

### 1.2 What is genuinely LLM-authored vs template-rendered

Per `content-generator.md:341-580`, the subagent must emit per-week:
4–5 `explanation`, 2–3 `example`, 2 `procedure`, 1–2 `real_world_scenario`,
1 `common_pitfall`, 1 `problem_solution`, 1 `self-check`, 1 `summary`, 1 `overview`
chunks (~15–18 chunks/week). Templates (`Courseforge/templates/chunk_templates.md`)
fix the HTML scaffold + `data-cf-*` attributes; the prose, examples, scenario
domains, deliverable text, misconception sentences, JSON-LD source-reference
arrays, key-term definitions, and assessment stems are LLM-authored. **Every one
of those surfaces is training data once Trainforge ingests the IMSCC.**

### 1.3 Decision capture today

`pipeline_tools.py:2774-2802` instantiates a `DecisionCapture` with
`phase="content-generator"` (canonical-enum, `decision_event.schema.json:53`)
and logs one `content_structure` event per run. There is no per-LLM-call
attribution event analogous to `synthesis_provider_call`
(`decision_event.schema.json:129`) or `curriculum_alignment_call`
(`decision_event.schema.json:84`). Phase 1 adds `content_generator_call`.

### 1.4 Reusable provider classes

All composition pieces already exist in `Trainforge/generators/`:
- `_anthropic_provider.py` — Anthropic SDK direct
- `_together_provider.py` — composes `OpenAICompatibleClient`
- `_local_provider.py` — composes `OpenAICompatibleClient`, supports Ollama/vLLM/llama.cpp/LM Studio
- `_openai_compatible_client.py:74-622` — JSON mode, retry/backoff, decision capture
- `_curriculum_provider.py:158-263` — the LLM-agnostic env-var-dispatch pattern to mirror

---

## 2. File-by-file changes

### 2.1 New file: `Courseforge/generators/__init__.py` (empty package marker)

### 2.2 New file: `Courseforge/generators/_provider.py` (~250 lines)

**Strategy:** Compose-and-reuse, NOT copy. Import `OpenAICompatibleClient` and
the per-provider env-var constants from the existing Trainforge modules so
one local server (Ollama on `:11434`) serves Courseforge, curriculum-alignment,
and synthesis simultaneously.

```python
# Constants
ENV_PROVIDER = "COURSEFORGE_PROVIDER"
DEFAULT_PROVIDER = "anthropic"   # backward compat; matches curriculum-alignment
SUPPORTED_PROVIDERS = ("anthropic", "together", "local")
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_TEMPERATURE = 0.4

# Class
class ContentGeneratorProvider:
    def __init__(
        self, *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capture: Optional[Any] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        client: Optional[Any] = None,             # test injection for OA-compatible
        anthropic_client: Optional[Any] = None,   # test injection for Anthropic
    ) -> None: ...

    def generate_page(
        self,
        *,
        course_code: str,
        week_number: int,
        page_id: str,
        page_template: str,           # the slotted HTML scaffold from Courseforge templates
        page_context: Dict[str, Any], # objectives, source blocks, prerequisite refs
    ) -> str:
        """Return the rendered HTML page (string for Phase 1; will return
        a Block in Phase 2). Emits one decision_type=content_generator_call
        event per call carrying provider/model/page_id/token usage."""
```

Provider-branch construction logic mirrors
`Trainforge/generators/_curriculum_provider.py:172-262` line-for-line:
- `anthropic` branch: lazy-import `anthropic`, model from `ANTHROPIC_SYNTHESIS_MODEL` or default, key from `ANTHROPIC_API_KEY`.
- `together` branch: model from `TOGETHER_SYNTHESIS_MODEL`, base URL `https://api.together.xyz/v1` (default), key from `TOGETHER_API_KEY`, instantiates `OpenAICompatibleClient(provider_label="together")`.
- `local` branch: model from `LOCAL_SYNTHESIS_MODEL`, base URL from `LOCAL_SYNTHESIS_BASE_URL` (default Ollama), key from `LOCAL_SYNTHESIS_API_KEY` or `"local"`, instantiates `OpenAICompatibleClient(provider_label="local")`.

`generate_page` builds a system prompt embedding the
`content-generator.md` core directives (Pattern 22 prevention, color palette,
Wave 79 chunk mix, source-grounding contract) and a user prompt carrying the
slotted template + `page_context` JSON. Routes through `_dispatch_call`
(anthropic SDK or `_oa_client._post_with_retry`); returns the assistant text
verbatim (Phase 1 — caller is responsible for HTML post-validation).

Decision-capture emit (after every dispatch):

```python
self._capture.log_decision(
    decision_type="content_generator_call",
    decision=(
        f"Courseforge content-generator authored {page_id} for "
        f"{course_code} week {week_number} via "
        f"provider={self._provider}, model={self._model}, "
        f"retry_count={retry_count}."
    ),
    rationale=(
        f"Routing content-generator authoring for course={course_code} "
        f"week={week_number} page={page_id} through provider="
        f"{self._provider}, model={self._model}"
        + (f", base_url={self._base_url}" if self._base_url else "")
        + f". Tokens prompt={prompt_toks} completion={comp_toks}; "
        f"retry_count={retry_count}. Backend choice operator-controlled "
        f"via COURSEFORGE_PROVIDER env (anthropic / together / local) "
        f"to keep Courseforge content-generator output ToS-clean for "
        f"Trainforge ingestion. See docs/LICENSING.md."
    ),
)
```

### 2.3 Wire-in: `MCP/tools/pipeline_tools.py::_generate_course_content`

Phase 1 wires the provider in two places:

1. **Dispatch-path interception (`pipeline_tools.py:2774-2802`, just before
   `capture.log_decision(content_structure, ...)`):** read
   `os.environ.get("COURSEFORGE_PROVIDER")`. If set, instantiate
   `ContentGeneratorProvider(capture=capture)` once at the start of the
   per-week loop and pass it down to `_gen.generate_week(...)` via a new
   kwarg `content_provider=...` (default `None`). When `None`, the existing
   deterministic-template path runs unchanged.

2. **`Courseforge/scripts/generate_course.py::generate_week` (~line TBD; locate
   the call site that today renders the slotted prose paragraphs):** when
   `content_provider` is non-None, swap the placeholder-fill step for
   `content_provider.generate_page(...)`. This is the surgical edit; the
   exact line is in the per-page rendering helper inside `generate_course.py`.

3. **Subagent path under `ED4ALL_AGENT_DISPATCH=true`:** the dispatch fork in
   `MCP/core/executor.py:823-868` short-circuits to `dispatcher.dispatch_task`
   BEFORE `_generate_course_content` runs. To make the provider switch fire
   under Wave 74 too, add a one-line guard at `executor.py:830` (just before
   the dispatch check):

   ```python
   if os.environ.get("COURSEFORGE_PROVIDER") and agent_type == "content-generator":
       # Force the in-process tool path so the provider switch fires; bypass
       # subagent dispatch even when ED4ALL_AGENT_DISPATCH=true. Operators
       # running a local provider don't want a Claude Code subagent shell.
       pass  # fall through to legacy in-process path
   else:
       # ... existing Wave 74 dispatch fork
   ```

   This is intentional: when the operator opts into a non-Anthropic provider,
   they want the provider, not the Claude session.

### 2.4 Schema: `schemas/events/decision_event.schema.json`

Add `"content_generator_call"` to the `decision_type` enum
(`decision_event.schema.json:63-130`). Sort lexically; mirror the trailing
comma style of the existing entries.

### 2.5 Documentation: `docs/LICENSING.md`

Two edits:

1. **Correct the false statement at line 28:**
   - Replace "Claude Code does not generate training data on this project" with:
     "Claude Code does not generate training data on this project EXCEPT through
     the Courseforge `content-generator` subagent when `ED4ALL_AGENT_DISPATCH=true`
     and `COURSEFORGE_PROVIDER` is unset. In that configuration, the subagent's
     Claude Code session authors HTML prose that Trainforge ingests as training
     chunks — i.e. it touches training data. Setting `COURSEFORGE_PROVIDER=local`
     (or `together`) routes the same surface through a license-clean provider.
     See the 'Synthesis providers' table below."

2. **Add a row to the providers table at line 46-60** (between
   `claude_session` and `together`):

   ```
   | `COURSEFORGE_PROVIDER=anthropic` | (same as `--provider anthropic`)
     | Anthropic proprietary | Anthropic Commercial Terms | **No** | Backward compat only |
   | `COURSEFORGE_PROVIDER=together` | (Together OSS) | Llama 3.3 / Qwen / DeepSeek
     | Together AI ToS | **Yes** | Hosted OSS fallback for content-generator |
   | `COURSEFORGE_PROVIDER=local`    | (Ollama / vLLM / llama.cpp / LM Studio) | Apache 2.0 (Qwen)
     | N/A | **Yes** | **Recommended** for ToS-clean Courseforge content |
   ```

3. **Add a section "Courseforge content-generator shares the synthesis provider stack"**
   immediately after the "Curriculum alignment shares the synthesis provider stack"
   section (line 83-87). One paragraph: explain that Courseforge content-generator
   reuses `LOCAL_SYNTHESIS_*` / `TOGETHER_SYNTHESIS_*` / `ANTHROPIC_SYNTHESIS_*`
   env vars and emits one `content_generator_call` decision event per page;
   recommend `local` for ToS-clean course generation.

### 2.6 Documentation: `CLAUDE.md` (root)

Add a row to the Opt-In Behavior Flags table (`CLAUDE.md:698-729`),
positioned alphabetically next to `CURRICULUM_ALIGNMENT_PROVIDER`:

```
| `COURSEFORGE_PROVIDER` | Selects the LLM backend for Courseforge content-generator
  page authoring (`Courseforge/generators/_provider.py::ContentGeneratorProvider`).
  Values: `anthropic` (legacy default — ToS-restricted for training-data),
  `together` (ToS-clean cloud OSS), `local` (recommended; 8GB-VRAM-friendly Qwen
  via Ollama). Reuses the same `TOGETHER_SYNTHESIS_*` / `LOCAL_SYNTHESIS_*` /
  `ANTHROPIC_SYNTHESIS_*` env vars as the synthesis pipeline so one local server
  serves Courseforge content-generator, curriculum-alignment, and Trainforge synthesis.
  When unset (default), Courseforge runs its deterministic template renderer; under
  `ED4ALL_AGENT_DISPATCH=true` content authoring routes through the `content-generator`
  subagent's own Claude Code session (Anthropic Consumer Terms — NOT recommended for
  training-data). Setting this env var ALSO short-circuits the Wave 74 dispatch fork
  so the provider drives authoring instead of the subagent. Captured per call in
  the `content_generator_call` decision event. See `docs/LICENSING.md`. |
```

### 2.7 Documentation: `Courseforge/CLAUDE.md`

Add a one-paragraph section "Provider selection (Phase 1 ToS unblock)"
near the "Quick Start" block, citing root `CLAUDE.md` and `docs/LICENSING.md`.
Do NOT duplicate the env-var table.

### 2.8 Test: `Courseforge/tests/test_content_generator_provider.py`

Mirror `Trainforge/tests/test_curriculum_alignment_provider.py:1-120`
test-by-test. Coverage:

- `test_unknown_provider_raises_value_error`
- `test_default_provider_is_anthropic_when_env_unset`
- `test_env_var_selects_provider` (sets `COURSEFORGE_PROVIDER=local`, asserts `_provider=="local"`)
- `test_local_provider_happy_path` (httpx.MockTransport returns a fake HTML page; assert `generate_page` returns it verbatim and one `content_generator_call` decision is logged)
- `test_together_provider_happy_path`
- `test_anthropic_provider_happy_path` (mock `anthropic.Anthropic` client)
- `test_decision_capture_fires_with_page_id_and_provider_in_rationale` (rationale ≥20 chars; contains course_code, week_number, page_id, provider, model)
- `test_pipeline_tools_routes_through_provider_when_env_set` (monkeypatch `COURSEFORGE_PROVIDER=local`, drive `_generate_course_content` with a stub `OpenAICompatibleClient`, assert no Anthropic call fires and the decision capture file contains a `content_generator_call` row)

Place tests under `Courseforge/tests/` (the existing `Courseforge/tests/`
directory hosts the cross-cutting Courseforge tests; check `pytest.ini` to
confirm collection). Mirror import style from the curriculum-alignment test
(`PROJECT_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, ...)`).

---

## 3. Decision-capture contract

| Field | Value |
|-------|-------|
| `decision_type` | `content_generator_call` (new enum entry) |
| `phase` | `content-generator` (already canonical, `decision_event.schema.json:53`) |
| `tool` | `courseforge` (already canonical, `decision_event.schema.json:58`) |
| `decision` | `"Courseforge content-generator authored {page_id} for {course_code} week {week_number} via provider={provider}, model={model}, retry_count={n}."` |
| `rationale` | ≥20 chars; interpolates `course_code`, `week_number`, `page_id`, `provider`, `model`, `base_url` (when set), prompt/completion token counts, retry count, plus the operator-control + ToS-routing justification. Sample template in §2.2 above. |
| `confidence` | omit (low signal — every emit is identical confidence) |
| `metadata` | `{"course_code": ..., "week_number": ..., "page_id": ..., "provider": ..., "model": ..., "tokens": {"prompt": p, "completion": c}}` |

Captures land at `training-captures/courseforge/<COURSE_CODE>/phase_content-generator/decisions_YYYYMMDD_HHMMSS.jsonl`
per `Trainforge/CLAUDE.md` "Decision Capture Output" convention.

---

## 4. Sequencing & dependencies

Execute in this order:

1. **Schema first:** add `content_generator_call` to `decision_event.schema.json`.
   (Independent; unblocks tests.)
2. **New module:** create `Courseforge/generators/_provider.py` + `__init__.py`.
   No callers yet; safe to land standalone.
3. **Tests:** add `Courseforge/tests/test_content_generator_provider.py`. Run them
   in isolation; no pipeline integration yet.
4. **Wire-in part 1:** `MCP/tools/pipeline_tools.py::_generate_course_content`
   reads `COURSEFORGE_PROVIDER` and instantiates the provider; passes it to
   `_gen.generate_week` as a new optional kwarg.
5. **Wire-in part 2:** `Courseforge/scripts/generate_course.py::generate_week`
   accepts the `content_provider` kwarg and calls `provider.generate_page(...)`
   for the LLM-authored page sections.
6. **Wave 74 short-circuit:** add the `executor.py:830`-area guard so
   `COURSEFORGE_PROVIDER` overrides the subagent dispatch fork.
7. **Documentation:** `docs/LICENSING.md` (correction + table row), root `CLAUDE.md`
   (env-var row), `Courseforge/CLAUDE.md` (one-line link).
8. **Pipeline integration test:** add the end-to-end test from §2.8 step 8 last,
   after every above step lands.

**Phase 2 dependency:** None. Phase 1's `generate_page` returns `str` (HTML).
When Phase 2 introduces `Block`, the return type changes; document in §2.2 with
a `# TODO Phase 2: return Block instead of str` comment, but do NOT block on it.

---

## 5. Validation plan

```bash
# Unit tests (offline, no network):
pytest Courseforge/tests/test_content_generator_provider.py -v

# End-to-end with local provider (operator):
export COURSEFORGE_PROVIDER=local
export LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1
export LOCAL_SYNTHESIS_MODEL=qwen2.5:14b-instruct-q4_K_M
ed4all run textbook_to_course \
  --inputs Courseforge/inputs/textbooks/<book>/ \
  --course-code DEMO_101 \
  --weeks 4

# Verify zero Anthropic in the decision-capture trail:
grep -l '"provider": "anthropic"' \
  training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl
# expected: no matches

# Verify provider attribution fired per page:
jq -r 'select(.decision_type=="content_generator_call") | .metadata.provider' \
  training-captures/courseforge/DEMO_101/phase_content-generator/decisions_*.jsonl
# expected: only "local"

# Verify the IMSCC packaged successfully:
ls Courseforge/exports/*/05_final_package/*.imscc
```

Acceptance criteria: zero Anthropic calls in the captures, every page emit
has a corresponding `content_generator_call` event, IMSCC validates per the
existing pipeline gates.

---

## 6. Risks & rollback

| Risk | Mitigation |
|------|------------|
| Local 14B-Q4 model produces malformed HTML that breaks downstream `package_imscc` validation. | `OpenAICompatibleClient`'s `json_mode=False` for HTML emission; add a one-page smoke fixture that exercises the full provider→IMSCC validator chain before kicking off a multi-week run. |
| `COURSEFORGE_PROVIDER` short-circuit at `executor.py:830` interacts badly with other agents that ARE supposed to dispatch. | Guard explicitly checks `agent_type == "content-generator"` so other Wave-74 agents (course-outliner, oscqr-course-evaluator, etc.) keep dispatching unchanged. |
| Decision-capture rationale interpolation drops below 20 chars on a degenerate input (empty `page_id`). | Validate inputs at `ContentGeneratorProvider.generate_page` entry; raise `ValueError` on empty page_id. Tests cover this. |
| `local` provider raises `ConnectionError` mid-run (Ollama crashed). | The provider's `OpenAICompatibleClient._post_with_retry` already retries; on exhaustion it raises `SynthesisProviderError`, which the caller's existing retry in `pipeline_tools.py` will surface as a task failure with retry. |

**Rollback:** unset `COURSEFORGE_PROVIDER`. The pre-Phase-1 path runs
unchanged — both the deterministic template renderer (default) and the
Wave 74 dispatch fork (when `ED4ALL_AGENT_DISPATCH=true`). The new module,
schema entry, and tests stay on disk; they are inert when the env var is
unset.

---

## 7. Open questions

1. Default value for `COURSEFORGE_PROVIDER` (`anthropic` for backward compat
   vs `local` to force ToS-clean by default).
2. Whether to ALSO expose a Phase-1 path that authors content WITHOUT the Wave 74
   subagent (i.e. always go through `ContentGeneratorProvider` even when
   `ED4ALL_AGENT_DISPATCH=false`), or strictly limit Phase 1 to "make the
   provider switch work when the operator wants it."
3. Whether to keep the `anthropic` enum value at all (curriculum-alignment kept
   it; one could argue Phase 1 should drop it on the Courseforge surface to
   avoid misuse).
4. Block dataclass shape (Phase 2 dependency) — `generate_page` returns `str`
   for Phase 1; confirm Phase 2 will adapt the call site without re-versioning
   the env var.
5. CLI flag (`--courseforge-provider`) on `ed4all run textbook_to_course` —
   curriculum-alignment has `--curriculum-provider`; do we want symmetry, or
   is env-var-only acceptable for Phase 1?
