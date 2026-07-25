"""``ed4all run --base-model`` — CLI surface, registry validation, resume, and
the campaign-harness argv pin.

Background (the LIVE bug this file exists to prevent recurring): the root
``CLAUDE.md``, a ``cli/commands/run.py`` help string, and — critically —
``lib/assistant/campaign_tools.launch_training_run`` all built/documented the
training invocation as::

    ed4all run trainforge_train --course-code <slug> --base-model <name>

``ed4all run`` declared NEITHER option. The campaign harness's Stage-B training
launch therefore died on a click ``UsageError`` before any workflow state was
created — a training path that was documented and wired but never executed
end-to-end. ``--course-code`` is the HANDLER-side param alias declared in
``config/workflows.yaml`` (``inputs_from: course_code <- course_name``), not a
CLI spelling.

Separately, ``config/workflows.yaml::training`` declares
``inputs_from: base_model <- workflow_params.base_model`` but nothing could set
``workflow_params.base_model``, so that route could only ever thread ``None``.
``--base-model`` makes it live.

The load-bearing test here is
:func:`test_campaign_training_argv_is_accepted_by_the_real_click_parser`: it
captures the argv ``launch_training_run`` ACTUALLY builds (by intercepting
``subprocess.Popen``) and feeds it through the REAL click command, rather than
eyeballing the string.
"""
from __future__ import annotations

import json
from typing import Dict, List

from click.testing import CliRunner

from cli.commands.run import (
    _apply_resume_base_model_override,
    _build_workflow_params,
    _validate_base_model,
)
from cli.main import cli


def _params(**overrides) -> dict:
    """``_build_workflow_params`` with its required kwargs filled in.

    The helper is keyword-only and takes the full CLI flag surface; this keeps
    the tests focused on the base-model threading.
    """
    base = dict(
        corpus=None,
        course_name="tst-101",
        weeks=None,
        no_assessments=False,
        assessment_count=50,
        bloom_levels="remember,understand,apply,analyze",
        priority="normal",
        objectives_path=None,
    )
    base.update(overrides)
    return _build_workflow_params("trainforge_train", **base)


#: A short name that MUST exist in Trainforge/training/base_models.py.
_VALID_BASE = "qwen2.5-1.5b"
_INVALID_BASE = "definitely-not-a-real-base"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
class TestOptionSurface:
    def test_base_model_option_is_declared(self):
        result = CliRunner().invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--base-model" in result.output

    def test_course_code_is_NOT_a_run_option(self):
        """``--course-code`` is a handler param alias, never a CLI spelling.

        Guards against 'fixing' the drift by minting a redundant second name
        for ``--course-name``.
        """
        result = CliRunner().invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--course-code" not in result.output

    def test_dry_run_accepts_course_name_and_base_model(self):
        result = CliRunner().invoke(
            cli,
            [
                "run", "trainforge_train",
                "--course-name", "tst-101",
                "--base-model", _VALID_BASE,
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        # The pin is surfaced, proving the flag reached workflow_params.
        assert _VALID_BASE in result.output

    def test_dry_run_json_carries_base_model_param(self):
        result = CliRunner().invoke(
            cli,
            [
                "run", "trainforge_train",
                "--course-name", "tst-101",
                "--base-model", _VALID_BASE,
                "--dry-run", "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads(result.output)
        assert plan["params"]["base_model"] == _VALID_BASE


# ---------------------------------------------------------------------------
# Registry validation — loud, never a silent substitution
# ---------------------------------------------------------------------------
class TestRegistryValidation:
    def test_unknown_base_model_exits_2_with_supported_list(self):
        result = CliRunner().invoke(
            cli,
            [
                "run", "trainforge_train",
                "--course-name", "tst-101",
                "--base-model", _INVALID_BASE,
                "--dry-run",
            ],
        )
        assert result.exit_code == 2, result.output
        assert _INVALID_BASE in result.output
        # The registry's own message enumerates every supported short name.
        assert "Supported bases" in result.output
        assert _VALID_BASE in result.output

    def test_unknown_base_model_never_substitutes_a_default(self):
        result = CliRunner().invoke(
            cli,
            [
                "run", "trainforge_train",
                "--course-name", "tst-101",
                "--base-model", _INVALID_BASE,
                "--dry-run",
            ],
        )
        # No plan is printed and no base is silently swapped in: the only
        # place a valid name may appear is inside the registry's own
        # "Supported bases: [...]" enumeration.
        assert "BaseModel:" not in result.output
        assert "Phases:" not in result.output
        head = result.output.split("Supported bases")[0]
        assert "nemotron3-nano-30b" not in head

    def test_validator_delegates_to_the_one_registry(self, monkeypatch):
        """No second, divergent allowlist: the check IS BaseModelRegistry."""
        import Trainforge.training.base_models as bm

        seen: List[str] = []
        original = bm.BaseModelRegistry.resolve

        def _spy(name):
            seen.append(name)
            return original(name)

        monkeypatch.setattr(bm.BaseModelRegistry, "resolve", staticmethod(_spy))
        _validate_base_model(_VALID_BASE)
        assert seen == [_VALID_BASE]

    def test_validator_is_a_noop_when_unset(self):
        # Unset must NOT pin anything — the handler's env/default chain governs.
        assert _validate_base_model(None) is None
        assert _validate_base_model("") is None


# ---------------------------------------------------------------------------
# Param threading + precedence (CLI > ED4ALL_CAMPAIGN_BASE_MODEL > default)
# ---------------------------------------------------------------------------
class TestParamThreading:
    def test_explicit_flag_lands_in_workflow_params(self):
        params = _params(base_model=_VALID_BASE)
        assert params["base_model"] == _VALID_BASE

    def test_unset_flag_omits_the_key_entirely(self):
        """Absent key => the handler falls through to env > registry default.

        Writing an explicit ``None`` would clobber that documented chain.
        """
        params = _params()
        assert "base_model" not in params

    def test_param_key_matches_the_yaml_inputs_from_route(self):
        """The whole point of the flag: make the declared route non-dead.

        ``config/workflows.yaml::training`` declares
        ``inputs_from: base_model <- workflow_params.base_model``. If the CLI
        wrote any other params key, the route would keep threading ``None``.
        """
        from MCP.core.workflow_runner import _get_phase_param_routing

        routing = _get_phase_param_routing("training")
        assert routing.get("base_model") == ("workflow_params", "base_model")
        assert routing.get("course_code") == ("workflow_params", "course_name")

        params = _params(base_model=_VALID_BASE)
        # Resolve the route by hand exactly as the runner does.
        _source, key = routing["base_model"]
        assert params.get(key) == _VALID_BASE
        _source, key = routing["course_code"]
        assert params.get(key) == "tst-101"

    def test_handler_precedence_chain_is_cli_then_env_then_default(
        self, monkeypatch,
    ):
        """CLI kwarg > ED4ALL_CAMPAIGN_BASE_MODEL > registry default.

        Exercised through the REAL registered ``run_training`` handler so there
        is one precedence implementation, not a test-local mirror. A bogus name
        is used so the handler short-circuits on ``unknown_base_model`` before
        touching disk/GPU — the envelope still reports the RESOLVED name.
        """
        from MCP.tools.pipeline_tools import _build_tool_registry

        run_training = _build_tool_registry()["run_training"]

        def _resolved(**kwargs) -> str:
            out = json.loads(_sync(run_training(course_name="tst-101", **kwargs)))
            return out["base_model"]

        monkeypatch.setenv("ED4ALL_CAMPAIGN_BASE_MODEL", "env-chosen-base")
        # CLI-routed kwarg wins over the env var.
        assert _resolved(base_model="cli-chosen-base") == "cli-chosen-base"
        # No kwarg -> env var.
        assert _resolved() == "env-chosen-base"
        # No kwarg, no env -> the campaign/registry default.
        monkeypatch.delenv("ED4ALL_CAMPAIGN_BASE_MODEL", raising=False)
        assert _resolved() == "nemotron3-nano-30b"


def _sync(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Resume — the flag must reach WorkflowRunner through persisted params
# ---------------------------------------------------------------------------
def _write_state(tmp_path, monkeypatch, params: dict) -> str:
    import lib.paths as paths_mod

    monkeypatch.setattr(paths_mod, "STATE_PATH", tmp_path / "state")
    workflow_id = "WF-BASEMODEL-TEST"
    wf_dir = tmp_path / "state" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps({"workflow_id": workflow_id, "params": params})
    )
    return workflow_id


def _read_params(tmp_path, workflow_id: str) -> dict:
    path = tmp_path / "state" / "workflows" / f"{workflow_id}.json"
    return json.loads(path.read_text())["params"]


class TestResumeOverride:
    def test_explicit_base_model_is_persisted(self, tmp_path, monkeypatch):
        wf_id = _write_state(tmp_path, monkeypatch, {"course_name": "tst-101"})
        assert _apply_resume_base_model_override(wf_id, _VALID_BASE) == _VALID_BASE
        assert _read_params(tmp_path, wf_id)["base_model"] == _VALID_BASE

    def test_explicit_base_model_repins_a_persisted_value(self, tmp_path, monkeypatch):
        wf_id = _write_state(
            tmp_path, monkeypatch, {"base_model": "nemotron3-nano-30b"},
        )
        _apply_resume_base_model_override(wf_id, _VALID_BASE)
        assert _read_params(tmp_path, wf_id)["base_model"] == _VALID_BASE

    def test_absent_flag_leaves_persisted_value_alone(self, tmp_path, monkeypatch):
        """Stickiness parity with --stop-after / --with-training."""
        wf_id = _write_state(tmp_path, monkeypatch, {"base_model": _VALID_BASE})
        assert _apply_resume_base_model_override(wf_id, None) is None
        assert _read_params(tmp_path, wf_id)["base_model"] == _VALID_BASE

    def test_unwritable_state_reports_loudly_and_returns_none(
        self, tmp_path, monkeypatch, capsys,
    ):
        import lib.paths as paths_mod

        monkeypatch.setattr(paths_mod, "STATE_PATH", tmp_path / "nowhere")
        assert _apply_resume_base_model_override("WF-MISSING", _VALID_BASE) is None
        err = capsys.readouterr().err
        assert "could NOT be persisted" in err

    def test_override_reuses_the_single_state_file_patcher(self, monkeypatch):
        """No third patcher — it must go through _persist_workflow_param."""
        import cli.commands.run as run_mod

        calls: List[tuple] = []
        monkeypatch.setattr(
            run_mod,
            "_persist_workflow_param",
            lambda wf, key, val: (calls.append((wf, key, val)), True)[1],
        )
        _apply_resume_base_model_override("WF-X", _VALID_BASE)
        assert calls == [("WF-X", "base_model", _VALID_BASE)]


class TestCreationPathPersist:
    """``create_textbook_pipeline`` has no base_model kwarg.

    The in-build ``--with-training`` tail would therefore ignore the flag
    entirely unless the creation path patches it onto the persisted params —
    the same gap ``with_training`` has, and the same fix.
    """

    def _create(self, monkeypatch, params: dict) -> List[tuple]:
        import cli.commands.run as run_mod
        import MCP.tools.pipeline_tools as pt

        patched: List[tuple] = []

        async def _fake_create(**kwargs):
            return json.dumps({"workflow_id": "WF-CREATE-TEST"})

        monkeypatch.setattr(pt, "create_textbook_pipeline", _fake_create)
        monkeypatch.setattr(
            run_mod,
            "_persist_workflow_param",
            lambda wf, key, val: (patched.append((wf, key, val)), True)[1],
        )
        _sync(run_mod._create_textbook_workflow(params))
        return patched

    def test_base_model_is_patched_onto_the_created_workflow(self, monkeypatch):
        patched = self._create(
            monkeypatch,
            {"course_name": "tst-101", "base_model": _VALID_BASE,
             "with_training": True},
        )
        assert ("WF-CREATE-TEST", "base_model", _VALID_BASE) in patched
        assert ("WF-CREATE-TEST", "with_training", True) in patched

    def test_absent_base_model_patches_nothing(self, monkeypatch):
        patched = self._create(monkeypatch, {"course_name": "tst-101"})
        assert not [row for row in patched if row[1] == "base_model"]


# ---------------------------------------------------------------------------
# THE PIN: the campaign harness's argv must parse against the REAL click
# option set. This is the test that would have caught the live bug.
# ---------------------------------------------------------------------------
def _capture_campaign_training_argv(monkeypatch) -> List[str]:
    """Run ``launch_training_run`` with its side effects stubbed, return argv.

    Everything that touches the world (preflight, docker seat teardown, the
    spawn itself, the WF-id poll, the launched-runs row, decision capture) is
    stubbed; the argv construction is the ONLY real code exercised.
    """
    import lib.assistant.campaign_tools as ct

    captured: Dict[str, List[str]] = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        return _FakeProc()

    monkeypatch.setattr(
        ct, "prepare_training_run", lambda slug: {"ok": True, "error": None},
    )
    monkeypatch.setattr(ct, "teardown_vllm_seats", lambda: None)
    monkeypatch.setattr(ct.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(ct, "_poll_training_wf_id", lambda _t: "WF-TEST")
    monkeypatch.setattr(ct, "_log_capture", lambda *a, **k: None)

    out = ct.launch_training_run("my-book")
    assert out["ok"] is True, out
    return captured["argv"]


class TestCampaignArgvPin:
    def test_campaign_training_argv_is_accepted_by_the_real_click_parser(
        self, monkeypatch,
    ):
        """Parse the ACTUAL argv the harness builds through the REAL command.

        Before the fix this asserted-through-click parse raised
        ``No such option: '--course-code'`` (exit 2), which is exactly what the
        campaign harness hit at launch time.
        """
        argv = _capture_campaign_training_argv(monkeypatch)

        assert argv[:3] == ["ed4all", "run", "trainforge_train"]
        # Drop the binary name; ``cli`` is the group, so keep "run" onward.
        result = CliRunner().invoke(cli, argv[1:] + ["--dry-run"])
        assert result.exit_code == 0, (
            f"campaign argv {argv!r} is NOT accepted by ed4all run:\n"
            f"{result.output}"
        )
        assert "my-book" in result.output

    def test_campaign_argv_uses_course_name_not_course_code(self, monkeypatch):
        argv = _capture_campaign_training_argv(monkeypatch)
        assert "--course-name" in argv
        assert "--course-code" not in argv

    def test_every_campaign_argv_flag_exists_on_the_run_command(self, monkeypatch):
        """Structural pin: each long flag must be a declared click option."""
        from cli.commands.run import run_command

        declared = {
            opt
            for param in run_command.params
            for opt in getattr(param, "opts", [])
            if opt.startswith("--")
        }
        argv = _capture_campaign_training_argv(monkeypatch)
        passed = [tok for tok in argv if tok.startswith("--")]
        assert passed, "campaign argv passed no flags — pin is vacuous"
        missing = sorted(set(passed) - declared)
        assert not missing, (
            f"campaign training argv passes flag(s) ed4all run does not "
            f"declare: {missing}. Declared: {sorted(declared)}"
        )

    def test_campaign_base_model_resolves_in_the_registry(self, monkeypatch):
        """The harness's base name must be a real registry entry."""
        from Trainforge.training.base_models import BaseModelRegistry
        import lib.assistant.campaign_tools as ct

        argv = _capture_campaign_training_argv(monkeypatch)
        base = argv[argv.index("--base-model") + 1]
        assert base == ct.resolve_campaign_base_model()
        BaseModelRegistry.resolve(base)  # raises KeyError if bogus
