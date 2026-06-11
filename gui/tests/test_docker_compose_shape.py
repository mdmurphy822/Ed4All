"""CI-friendly lint over the deployable Docker artifacts.

Parses ``docker-compose.yml`` and ``Dockerfile.gui`` with no Docker engine and
asserts the shape that ``docker compose up`` on a clean machine depends on, so
the deploy story can't drift silently. Pure ``yaml.safe_load`` / text checks —
no network, no heavy deps.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
DOCKERFILE_GUI = REPO_ROOT / "Dockerfile.gui"


def _load_compose() -> dict:
    with COMPOSE_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_compose_file_parses():
    doc = _load_compose()
    assert isinstance(doc, dict)
    assert "services" in doc and isinstance(doc["services"], dict)


def test_gui_service_builds_from_dockerfile_gui():
    gui = _load_compose()["services"]["gui"]
    # Must have a build key (image is built from the repo, not pulled).
    assert "build" in gui, "gui service must build from the repo context"
    build = gui["build"]
    assert isinstance(build, dict)
    assert build.get("dockerfile") == "Dockerfile.gui"
    assert build.get("context") == "."


def test_gui_service_has_healthcheck_on_health_endpoint():
    gui = _load_compose()["services"]["gui"]
    hc = gui.get("healthcheck")
    assert hc, "gui service must declare a healthcheck"
    test = hc["test"]
    joined = " ".join(test) if isinstance(test, list) else str(test)
    assert "/api/health" in joined


def test_gui_service_sets_ed4all_home():
    gui = _load_compose()["services"]["gui"]
    env = gui.get("environment", {})
    assert env.get("ED4ALL_HOME") == "/data"
    # Studio is the deployable serve mode.
    assert env.get("ED4ALL_GUI_MODE") == "studio"


def test_gui_service_does_not_bind_mount_the_repo():
    """Mounting the repo over /app would shadow the baked-in code."""
    gui = _load_compose()["services"]["gui"]
    for vol in gui.get("volumes", []):
        spec = vol if isinstance(vol, str) else f"{vol.get('source')}:{vol.get('target')}"
        source = spec.split(":", 1)[0]
        # No bind mount of the repo root or its code dirs into the container.
        assert source not in {".", "./", "..", REPO_ROOT.as_posix()}
        assert not source.startswith("./lib")
        assert not source.startswith("./gui")
        assert not source.startswith("./MCP")
    # The data volume IS mounted at /data (= ED4ALL_HOME).
    targets = [
        (v.split(":")[1] if isinstance(v, str) and ":" in v else None)
        for v in gui.get("volumes", [])
    ]
    assert "/data" in targets


def test_loopback_policy_uses_host_networking_for_both_services():
    """The answer path requires loopback; host networking is how the default
    compose keeps GUI<->Ollama on localhost without weakening the policy."""
    services = _load_compose()["services"]
    assert services["gui"].get("network_mode") == "host"
    assert services["ollama"].get("network_mode") == "host"


def test_named_volumes_declared():
    doc = _load_compose()
    volumes = doc.get("volumes", {})
    assert "ed4all-data" in volumes
    assert "ollama-models" in volumes


def test_dockerfile_gui_exposes_port_and_pins_studio_data_env():
    text = DOCKERFILE_GUI.read_text(encoding="utf-8")
    assert "EXPOSE 8077" in text
    assert "ED4ALL_HOME=/data" in text
    assert "ED4ALL_GUI_MODE=studio" in text
    assert "HF_HOME=" in text  # embedding cache persisted onto the volume
    assert "HEALTHCHECK" in text
    assert "/api/health" in text
    # Installs the two extras the GUI needs.
    assert "[gui,server]" in text
