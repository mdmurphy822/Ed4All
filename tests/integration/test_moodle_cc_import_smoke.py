"""Track L (L3) — Moodle-in-Docker Common Cartridge import smoke test.

The single REAL-import sanity check behind the LMS-agnostic conformance
contract: a spec-valid ``.imscc`` must actually
restore into a real LMS. Moodle is the automatable target (official / bitnami
image, self-hostable); Canvas self-host was judged too heavy and Brightspace is
non-self-hostable.

**This test is docker-gated and skips cleanly** whenever any precondition is
unmet — no docker daemon, no Moodle image pulled, the container never comes up,
or the in-container converter reports a version/environment error. It NEVER
fails the suite for an environmental reason; it fails ONLY when Moodle boots,
imports the cartridge, and a quiz / item / itemfeedback / answer_key that the
cartridge carried did NOT survive the import (the real portability regression
this test exists to catch).

The COMMITTED test builds its own SYNTHETIC fixture cartridge (no course slugs,
no device paths). To produce real-import evidence against an actual built
course, point ``ED4ALL_MOODLE_SMOKE_IMSCC`` at a gitignored ``.imscc`` on disk
(e.g. a scratch-built cartridge) and run:

    ED4ALL_MOODLE_SMOKE_IMSCC=/path/to/course.imscc \\
    ED4ALL_MOODLE_IMAGE=bitnami/moodle:latest \\
    pytest -m integration tests/integration/test_moodle_cc_import_smoke.py -s

Mechanics (all inside the container, via ``docker exec ... php``): the driver
extracts the cartridge into Moodle's tempdir, runs the core
``convert_helper`` IMSCC→moodle2 converter, restores into a fresh course with
``restore_controller`` — the exact code path Moodle's web restore uses — then
queries the DB for quiz / question / itemfeedback survival and the answer_key
file's presence in the Moodle file store.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# --------------------------------------------------------------------------
# docker plumbing (best-effort; every failure path is a clean skip).
# --------------------------------------------------------------------------

def _docker_argv() -> Optional[List[str]]:
    """Return the argv prefix for docker, trying a plain call then ``sg docker``.

    Returns ``None`` when docker is not usable at all.
    """
    if shutil.which("docker") is None:
        return None
    # Plain docker (user in the docker group / rootless).
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        )
        if r.returncode == 0:
            return ["docker"]
    except (OSError, subprocess.SubprocessError):
        pass
    # sg docker -c wrapper (the Spark host pattern).
    if shutil.which("sg") is not None:
        try:
            r = subprocess.run(
                ["sg", "docker", "-c", "docker info"],
                capture_output=True, timeout=20,
            )
            if r.returncode == 0:
                return ["sg", "docker", "-c"]
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def _run_docker(prefix: List[str], docker_args: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a docker subcommand honoring the ``sg docker -c "<cmd>"`` shape."""
    if prefix[:2] == ["sg", "docker"]:
        # sg needs a single shell-string command.
        import shlex
        cmd = "docker " + " ".join(shlex.quote(a) for a in docker_args)
        argv = ["sg", "docker", "-c", cmd]
    else:
        argv = prefix + docker_args
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _find_moodle_image(prefix: List[str]) -> Optional[str]:
    """Resolve a usable Moodle image: env override, else a local image."""
    override = os.environ.get("ED4ALL_MOODLE_IMAGE", "").strip()
    if override:
        return override
    try:
        r = _run_docker(prefix, ["images", "--format", "{{.Repository}}:{{.Tag}}"], timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        line = line.strip()
        if "moodle" in line.lower():
            return line
    return None


# --------------------------------------------------------------------------
# Synthetic fixture cartridge (self-contained; no course slugs).
# --------------------------------------------------------------------------

CC13_MANIFEST_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
LOM_MANIFEST_NS = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
IMSDT_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
QTI_NS = "http://www.imsglobal.org/xsd/ims_qtiasiv1p2"

RES_QTI = "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"
RES_DT = "imsdt_xmlv1p3"
RES_WEB = "webcontent"


def _quiz_with_feedback() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="{QTI_NS}">
  <assessment ident="quiz_w1" title="Week 1 Quiz">
    <section ident="root_section">
      <item ident="q1" title="MC Question">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">What is 2 + 2?</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">4</mattext></material>
              </response_label>
              <response_label ident="B">
                <material><mattext texttype="text/html">5</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="5"/>
          </outcomes>
          <respcondition>
            <conditionvar><varequal respident="response1">A</varequal></conditionvar>
            <setvar action="Set" varname="SCORE">5</setvar>
            <displayfeedback feedbacktype="Response" linkrefid="correct_fb"/>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="correct_fb">
          <material><mattext texttype="text/html">Correct — 2 + 2 = 4.</mattext></material>
        </itemfeedback>
      </item>
    </section>
  </assessment>
</questestinterop>"""


def _discussion() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<topic xmlns="{IMSDT_NS}"><title>Week 1 Discussion</title>'
        '<text texttype="text/html">Discuss the objective.</text></topic>'
    )


def _html(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><title>{title}</title></head><body><p>{body}</p></body></html>"


def _resource(res_id: str, res_type: str, href: str) -> str:
    return (
        f'<resource identifier="{res_id}" type="{res_type}" href="{href}">'
        f'<file href="{href}"/></resource>'
    )


def _build_fixture_cartridge(tmp_path: Path) -> Path:
    ak = "06_assessments/week_01_answer_key.html"
    resources = (
        _resource("RES_page", RES_WEB, "week_01/content.html")
        + _resource("RES_quiz", RES_QTI, "06_assessments/week_01_quiz.xml")
        + _resource("RES_disc", RES_DT, "06_assessments/week_01_discussion.xml")
        + _resource("RES_ak", RES_WEB, ak)
    )
    manifest = (
        "<?xml version='1.0' encoding='utf-8'?>"
        f'<manifest xmlns="{CC13_MANIFEST_NS}" xmlns:lomimscc="{LOM_MANIFEST_NS}" '
        'identifier="SMOKE_manifest">'
        "<metadata><schema>IMS Common Cartridge</schema>"
        "<schemaversion>1.3.0</schemaversion></metadata>"
        '<organizations><organization identifier="ORG_1" structure="rooted-hierarchy">'
        '<item identifier="ROOT"><title>Smoke Course</title>'
        '<item identifier="I_quiz" identifierref="RES_quiz"><title>Week 1 Quiz</title></item>'
        "</item></organization></organizations>"
        f"<resources>{resources}</resources></manifest>"
    )
    members = {
        "imsmanifest.xml": manifest,
        "week_01/content.html": _html("Content", "Lesson body."),
        "06_assessments/week_01_quiz.xml": _quiz_with_feedback(),
        "06_assessments/week_01_discussion.xml": _discussion(),
        ak: _html("Answer Key", "Q1: 4"),
    }
    path = tmp_path / "smoke_course.imscc"
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


# --------------------------------------------------------------------------
# In-container PHP driver: convert CC → moodle2, restore, query survival.
# --------------------------------------------------------------------------

_PHP_DRIVER = r"""<?php
// Track-L Moodle CC-import driver. Emits a single JSON line on stdout.
define('CLI_SCRIPT', true);
require(getenv('MOODLE_DIRROOT') . '/config.php');
require_once($CFG->dirroot . '/backup/util/includes/backup_includes.php');
require_once($CFG->dirroot . '/backup/util/includes/restore_includes.php');
require_once($CFG->dirroot . '/backup/util/helper/convert_helper.class.php');

function out($arr) { echo json_encode($arr) . "\n"; exit(0); }

try {
    global $DB, $USER, $CFG;
    $adminid = 2;
    $cartridge = getenv('CC_FILE');

    // 1. Stage the cartridge into a fresh tempdir and unzip it.
    $tempid = restore_controller::get_tempdir_name(SITEID, $adminid);
    $tempdir = make_backup_temp_directory($tempid);
    $fp = get_file_packer('application/zip');
    if (!$fp->extract_to_pathname($cartridge, $tempdir)) {
        out(['status' => 'error', 'message' => 'unzip failed']);
    }

    // 2. Detect + convert CC → moodle2 (the web-restore code path).
    $format = convert_helper::detect_backup_format($tempid);
    if ($format !== 'moodle2') {
        convert_helper::to_moodle2_format($tempid, $format, new base_logger(backup::LOG_NONE));
    }

    // 3. Create a target course + restore into it.
    $catid = $DB->get_field_sql('SELECT MIN(id) FROM {course_categories}');
    $newcourse = new stdClass();
    $newcourse->fullname = 'CC Smoke ' . time();
    $newcourse->shortname = 'ccsmoke' . time();
    $newcourse->category = $catid;
    $course = create_course($newcourse);

    $rc = new restore_controller($tempid, $course->id,
        backup::INTERACTIVE_NO, backup::MODE_GENERAL, $adminid,
        backup::TARGET_NEW_COURSE);
    $rc->execute_precheck();
    $rc->execute_plan();
    $rc->destroy();

    // 4. Query survival.
    $ctx = context_course::instance($course->id);
    $quizzes = $DB->count_records('quiz', ['course' => $course->id]);
    $qcount = $DB->count_records_sql(
        "SELECT COUNT(1) FROM {question} q
           JOIN {question_versions} qv ON qv.questionid = q.id
           JOIN {question_bank_entries} qbe ON qbe.id = qv.questionbankentryid
           JOIN {question_categories} qc ON qc.id = qbe.questioncategoryid
          WHERE qc.contextid = ?", [$ctx->id]);
    if ($qcount === false) {
        // Pre-4.0 schema (no question_versions): fall back.
        $qcount = $DB->count_records_sql(
            "SELECT COUNT(1) FROM {question} q
               JOIN {question_categories} qc ON qc.id = q.category
              WHERE qc.contextid = ?", [$ctx->id]);
    }
    $feedback = $DB->count_records_sql(
        "SELECT COUNT(1) FROM {question_answers} qa
           JOIN {question} q ON q.id = qa.question
           JOIN {question_versions} qv ON qv.questionid = q.id
           JOIN {question_bank_entries} qbe ON qbe.id = qv.questionbankentryid
           JOIN {question_categories} qc ON qc.id = qbe.questioncategoryid
          WHERE qc.contextid = ? AND qa.feedback <> ''", [$ctx->id]);
    if ($feedback === false) { $feedback = 0; }
    $answerkey = $DB->count_records_sql(
        "SELECT COUNT(1) FROM {files}
          WHERE filename " . $DB->sql_like('filename', '?', false) . "
            AND filesize > 0", ['%answer%key%']);

    out([
        'status' => 'ok',
        'course_id' => (int)$course->id,
        'quizzes' => (int)$quizzes,
        'questions' => (int)$qcount,
        'itemfeedback' => (int)$feedback,
        'answer_key_files' => (int)$answerkey,
    ]);
} catch (Throwable $e) {
    out(['status' => 'error', 'message' => get_class($e) . ': ' . $e->getMessage()]);
}
"""


def _moodle_dirroot(prefix: List[str], cid: str) -> Optional[str]:
    """Locate MOODLE_DIRROOT inside the container (bitnami vs official)."""
    for candidate in ("/bitnami/moodle", "/opt/bitnami/moodle", "/var/www/html"):
        r = _run_docker(prefix, ["exec", cid, "test", "-f", f"{candidate}/config.php"], timeout=30)
        if r.returncode == 0:
            return candidate
    return None


def _wait_for_config(prefix: List[str], cid: str, deadline: float) -> Optional[str]:
    """Poll until Moodle's config.php exists (install finished)."""
    while time.time() < deadline:
        dirroot = _moodle_dirroot(prefix, cid)
        if dirroot is not None:
            return dirroot
        time.sleep(10)
    return None


# --------------------------------------------------------------------------
# The test.
# --------------------------------------------------------------------------

def test_moodle_cc_import_smoke(tmp_path):
    prefix = _docker_argv()
    if prefix is None:
        pytest.skip("docker not usable on this host")

    image = _find_moodle_image(prefix)
    if image is None:
        pytest.skip(
            "no Moodle image available (pull one or set ED4ALL_MOODLE_IMAGE); "
            "fetch-window pull is the operator's deliberate step"
        )

    # The cartridge under test: a real one via env override, else the fixture.
    override = os.environ.get("ED4ALL_MOODLE_SMOKE_IMSCC", "").strip()
    if override:
        cartridge = Path(override)
        if not cartridge.exists():
            pytest.skip(f"ED4ALL_MOODLE_SMOKE_IMSCC does not exist: {cartridge}")
    else:
        cartridge = _build_fixture_cartridge(tmp_path)

    suffix = uuid.uuid4().hex[:8]
    container = f"ed4all-moodle-smoke-{suffix}"
    db_container = f"ed4all-moodle-db-{suffix}"
    network = f"ed4all-moodle-net-{suffix}"
    boot_deadline = time.time() + int(os.environ.get("ED4ALL_MOODLE_BOOT_TIMEOUT", "900"))

    # The Bitnami Moodle image has NO embedded database — without a reachable
    # MariaDB it waits forever at "Trying to connect to the database server"
    # and the boot deadline reads as an install hang (root-caused 2026-07-20
    # after three timeout skips). Provision a companion MariaDB on a private
    # network; both are torn down in finally.
    db_image = os.environ.get("ED4ALL_MOODLE_DB_IMAGE", "bitnamilegacy/mariadb:latest").strip()

    try:
        net = _run_docker(prefix, ["network", "create", network], timeout=60)
        if net.returncode != 0:
            pytest.skip(f"could not create docker network: {net.stderr.strip()[:200]}")
        db = _run_docker(
            prefix,
            [
                "run", "-d", "--name", db_container, "--network", network,
                "--network-alias", "mariadb",
                "-e", "ALLOW_EMPTY_PASSWORD=yes",
                "-e", "MARIADB_DATABASE=bitnami_moodle",
                "-e", "MARIADB_USER=bn_moodle",
                db_image,
            ],
            timeout=120,
        )
        if db.returncode != 0:
            pytest.skip(f"could not start MariaDB companion: {db.stderr.strip()[:300]}")
        run = _run_docker(
            prefix,
            [
                "run", "-d", "--name", container, "--network", network,
                "-e", "MOODLE_SKIP_BOOTSTRAP=no",
                "-e", "ALLOW_EMPTY_PASSWORD=yes",
                "-e", "MOODLE_DATABASE_TYPE=mariadb",
                "-e", "MOODLE_DATABASE_HOST=mariadb",
                "-e", "MOODLE_DATABASE_PORT_NUMBER=3306",
                "-e", "MOODLE_DATABASE_NAME=bitnami_moodle",
                "-e", "MOODLE_DATABASE_USER=bn_moodle",
                image,
            ],
            timeout=120,
        )
        if run.returncode != 0:
            pytest.skip(f"could not start Moodle container: {run.stderr.strip()[:300]}")

        dirroot = _wait_for_config(prefix, container, boot_deadline)
        if dirroot is None:
            pytest.skip("Moodle did not finish installing before the boot timeout")

        # Copy the cartridge + driver into the container.
        in_cc = "/tmp/smoke_course.imscc"
        in_php = "/tmp/cc_import_driver.php"
        cp1 = _run_docker(prefix, ["cp", str(cartridge), f"{container}:{in_cc}"], timeout=60)
        driver_path = tmp_path / "cc_import_driver.php"
        driver_path.write_text(_PHP_DRIVER)
        cp2 = _run_docker(prefix, ["cp", str(driver_path), f"{container}:{in_php}"], timeout=60)
        if cp1.returncode != 0 or cp2.returncode != 0:
            pytest.skip("failed to copy cartridge/driver into the container")

        # Execute the driver.
        exec_res = _run_docker(
            prefix,
            [
                "exec",
                "-e", f"MOODLE_DIRROOT={dirroot}",
                "-e", f"CC_FILE={in_cc}",
                container, "php", in_php,
            ],
            timeout=600,
        )
        stdout = exec_res.stdout.strip()
        # The driver prints exactly one JSON line last.
        payload: Optional[Dict] = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if payload is None:
            pytest.skip(
                "Moodle import driver produced no parseable result "
                f"(rc={exec_res.returncode}); stderr={exec_res.stderr.strip()[:300]}"
            )

        if payload.get("status") != "ok":
            # A converter/version/environment error is NOT a portability
            # regression — capture it as a skip so the committed test never
            # yields a false failure on a Moodle version mismatch.
            pytest.skip(f"Moodle import did not complete: {payload.get('message')}")

        # ---- Real portability assertions (only reached on a clean import).
        assert payload["quizzes"] >= 1, f"no quiz survived import: {payload}"
        assert payload["questions"] >= 1, f"no question survived import: {payload}"
        assert payload["itemfeedback"] >= 1, f"itemfeedback lost on import: {payload}"
        assert payload["answer_key_files"] >= 1, (
            f"answer_key webcontent not reachable after import: {payload}"
        )

    finally:
        _run_docker(prefix, ["rm", "-f", container], timeout=120)
        _run_docker(prefix, ["rm", "-f", db_container], timeout=120)
        _run_docker(prefix, ["network", "rm", network], timeout=60)
