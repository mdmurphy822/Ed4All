#!/usr/bin/env python3
"""One-command OpenOLAT provisioning for the Ed4All retrieval demo.

Builds, over the OpenOLAT REST API (no manual clicks), the course shell that
hosts the grounded-ask widget:

  1. a published course (find-or-create, keyed by externalRef=<slug>);
  2. the unpacked cartridge reading content as native **Single Page** course
     elements, one structure node per week (the CC HTML pages — the ONLY part
     of the .imscc that OpenOLAT can consume; see FIDELITY.md);
  3. the **External Page** course element that iframe-embeds the Ed4All learner
     SPA at  <widget-base>/learn/?course=<slug>&embed=1  (the demo's star);
  4. a demo learner user (find-or-create).

Everything is stdlib-only (urllib) so it runs anywhere Python 3.9+ exists — no
pip install. The cartridge's QTI 1.2 quizzes / imsdt discussions are
deliberately NOT imported (OpenOLAT 19.x cannot read them — see FIDELITY.md).

Usage:
  python3 provision.py \
      --olat-base   http://127.0.0.1:8080/openolat \
      --admin-user  administrator --admin-pass openolat \
      --widget-base http://localhost:8077 \
      --slug <course-slug>        # or set ED4ALL_DEMO_SLUG (gitignored .env)
      # --cartridge defaults to LibV2/courses/<slug>/source/imscc/<slug>.imscc

Admin credentials come from --admin-pass or the OLAT_ADMIN_PASS env var; nothing
secret is written to disk. Re-runnable: an existing course (same externalRef) is
reused unless --force creates a fresh one.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

# ------------------------------------------------------------------ HTTP helpers


class Rest:
    def __init__(self, base: str, user: str, password: str):
        self.base = base.rstrip("/") + "/restapi"
        tok = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.auth = f"Basic {tok}"

    def _open(self, req: urllib.request.Request, timeout: int = 120):
        req.add_header("Authorization", self.auth)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise SystemExit(
                f"REST {req.get_method()} {req.full_url} -> HTTP {exc.code}\n{body}"
            )
        except urllib.error.URLError as exc:
            raise SystemExit(f"REST {req.full_url} unreachable: {exc}")

    def call(self, method: str, path: str, params=None, accept="application/json",
             timeout: int = 120):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items()
                                                 if v is not None})
        req = urllib.request.Request(url, method=method)
        req.add_header("Accept", accept)
        resp = self._open(req, timeout=timeout)
        raw = resp.read().decode("utf-8", "replace")
        if not raw.strip():
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def put_json(self, path, obj, params=None, timeout: int = 120):
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(obj).encode()
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        resp = self._open(req, timeout=timeout)
        raw = resp.read().decode("utf-8", "replace")
        return json.loads(raw) if raw.strip() else None

    def put_multipart_file(self, path, params, field_file: Path, filename: str,
                           timeout: int = 120):
        """PUT a multipart/form-data body carrying one file part + a filename
        field (the shape OpenOLAT's attachSinglePage / coursefolder expect)."""
        url = self.base + path + "?" + urllib.parse.urlencode(params)
        boundary = "----ed4all" + uuid.uuid4().hex
        payload = field_file.read_bytes()
        parts = []
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="filename"\r\n\r\n')
        parts.append(filename.encode() + b"\r\n")
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            .encode()
        )
        parts.append(b"Content-Type: text/html\r\n\r\n")
        parts.append(payload + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        req = urllib.request.Request(url, data=body, method="PUT")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Accept", "application/json")
        resp = self._open(req, timeout=timeout)
        raw = resp.read().decode("utf-8", "replace")
        return json.loads(raw) if raw.strip() else None


# ------------------------------------------------------------------ steps


def find_course(rest: Rest, external_ref: str):
    courses = rest.call("GET", "/repo/courses", {"start": 0, "limit": 500}) or []
    if isinstance(courses, dict):  # some builds wrap in {courses:[...]}
        courses = courses.get("courses", [])
    for c in courses:
        if isinstance(c, dict) and c.get("externalRef") == external_ref:
            return c
    return None


def create_course(rest: Rest, slug: str, title: str):
    c = rest.call("PUT", "/repo/courses", {
        "shortTitle": slug[:25],
        "title": title,
        "displayName": title,
        "externalRef": slug,
        "status": "published",
    })
    return c


def add_structure(rest: Rest, cid, parent, short, long_):
    node = rest.call("PUT", f"/repo/courses/{cid}/elements/structure", {
        "parentNodeId": parent, "shortTitle": short[:25], "longTitle": long_,
    })
    return node["id"]


def add_singlepage(rest: Rest, cid, parent, short, long_, html_file: Path):
    node = rest.put_multipart_file(
        f"/repo/courses/{cid}/elements/singlepage",
        {"parentNodeId": parent, "shortTitle": short[:25], "longTitle": long_,
         "filename": html_file.name},
        html_file, html_file.name,
    )
    return node["id"]


def add_externalpage(rest: Rest, cid, parent, short, long_, url):
    node = rest.call("PUT", f"/repo/courses/{cid}/elements/externalpage", {
        "parentNodeId": parent, "shortTitle": short[:25], "longTitle": long_,
        "url": url,
    })
    return node["id"]


def create_learner(rest: Rest, login, password, email):
    # find-or-create by login via the user search (GET /users/{key} needs a
    # NUMERIC key, so search by the login query param instead).
    hits = rest.call("GET", "/users", {"login": login}) or []
    if isinstance(hits, list) and hits and hits[0].get("key"):
        return hits[0], False
    user = rest.put_json("/users", {
        "login": login, "password": password,
        "firstName": "Demo", "lastName": "Learner",
        "email": email,
    })
    return user, True


PRETTY = {
    "overview": "Overview", "summary": "Summary", "self_check": "Self Check",
    "application": "Application",
}


def page_title(fname: str) -> str:
    stem = Path(fname).stem  # week_01_content_02
    for k, v in PRETTY.items():
        if stem.endswith(k):
            return v
    if "_content_" in stem:
        n = stem.split("_content_")[-1]
        return f"Content {n}"
    if stem == "learning_objectives":
        return "Learning Objectives"
    return stem.replace("_", " ").title()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--olat-base", default=os.environ.get(
        "OLAT_BASE", "http://127.0.0.1:8080/openolat"))
    ap.add_argument("--admin-user", default=os.environ.get("OLAT_ADMIN_USER",
                                                           "administrator"))
    ap.add_argument("--admin-pass", default=os.environ.get("OLAT_ADMIN_PASS",
                                                           "openolat"))
    ap.add_argument("--widget-base", default=os.environ.get(
        "ED4ALL_WIDGET_BASE", "http://localhost:8077"))
    # Course slug is machine-local demo data, never hardcoded in tracked code:
    # pass --slug or set ED4ALL_DEMO_SLUG (e.g. from the gitignored .env). The
    # cartridge path defaults, relative to the slug, to the standard LibV2 layout.
    ap.add_argument("--slug", default=os.environ.get("ED4ALL_DEMO_SLUG"))
    ap.add_argument("--cartridge", default=os.environ.get("ED4ALL_DEMO_CARTRIDGE"))
    ap.add_argument("--learner-login", default="demo-learner")
    ap.add_argument("--learner-pass", default=os.environ.get(
        "OLAT_LEARNER_PASS", "demo-learner-pw"))
    ap.add_argument("--learner-email", default=os.environ.get(
        "OLAT_LEARNER_EMAIL", "demo-learner@example.com"))
    ap.add_argument("--force", action="store_true",
                    help="create a fresh course even if one with this externalRef exists")
    ap.add_argument("--max-weeks", type=int, default=0,
                    help="import only the first N weeks (0 = all); the widget is added regardless")
    args = ap.parse_args()

    if not args.slug:
        ap.error("no course slug: pass --slug <slug> or set ED4ALL_DEMO_SLUG "
                 "(see demo.env.example / the gitignored .env)")
    if not args.cartridge:
        args.cartridge = str(
            Path(__file__).resolve().parents[2]
            / f"LibV2/courses/{args.slug}/source/imscc/{args.slug}.imscc")
    if not Path(args.cartridge).is_file():
        ap.error(f"cartridge not found: {args.cartridge}")

    rest = Rest(args.olat_base, args.admin_user, args.admin_pass)

    # 0. auth sanity (the version endpoint emits text/plain only)
    ver = rest.call("GET", "/users/version", accept="*/*")
    print(f"[ok ] REST reachable, users API version {ver}")

    # 1. course (find-or-create). Content + widget are added only on a FRESH
    # course (or --force) so a re-run is a safe no-op that still ensures the
    # learner — the element-attach REST calls are not themselves idempotent.
    existing = None if args.force else find_course(rest, args.slug)
    fresh = existing is None
    if existing:
        course = existing
        cid = course["key"]
        print(f"[reuse] course externalRef={args.slug} key={cid} "
              f"(already provisioned; skipping content+widget — use --force to rebuild)")
    else:
        title = f"{args.slug} — Elementary Algebra (Ed4All retrieval demo)"
        course = create_course(rest, args.slug, title)
        cid = course["key"]
        print(f"[new ] course key={cid} status={course.get('repoEntryStatus')}")
    root = rest.call("GET", f"/repo/courses/{cid}")["editorRootNodeId"]
    # `?course=<slug>` pins+disables the course picker; `&embed=1` is the
    # EMBED-track compact-widget contract (hides picker + Quizzes panel — see
    # gui/static/learn/learn.js). Both together give the framed compact widget.
    widget_url = f"{args.widget_base.rstrip('/')}/learn/?course={args.slug}&embed=1"

    n_pages = 0
    weeks = []
    if fresh:
        # 2. widget FIRST (the demo's star element)
        wid = add_externalpage(rest, cid, root, "Ask the Course",
                               "Ask the Course (Ed4All grounded answers)", widget_url)
        print(f"[ok ] External Page (widget) node={wid} -> {widget_url}")

        # 3. reading content from the cartridge
        n_pages, weeks = _import_content(rest, cid, root, args)
        print(f"[ok ] imported {n_pages} reading pages across "
              f"{len(weeks)} week module(s)")

    # 4. learner (always ensured — idempotent)
    user, created = create_learner(rest, args.learner_login, args.learner_pass,
                                   args.learner_email)
    if user:
        print(f"[{'new ' if created else 'reuse'}] learner login={args.learner_login} "
              f"key={user.get('key')}")

    print("\n=== DONE ===")
    print(f"Course run URL : {args.olat_base.rstrip('/')}/url/RepositoryEntry/{cid}")
    print(f"Admin console  : {args.olat_base.rstrip('/')}/  (administrator / <admin-pass>)")
    print(f"Widget element : External Page -> {widget_url}")
    print(f"Learner login  : {args.learner_login} / <learner-pass>")


def _import_content(rest: Rest, cid, root, args):
    n_pages = 0
    weeks = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        with zipfile.ZipFile(args.cartridge) as zf:
            zf.extractall(tdp)

        # course_overview / learning objectives (if present)
        lo = tdp / "course_overview" / "learning_objectives.html"
        if lo.exists():
            add_singlepage(rest, cid, root, "Learning Objectives",
                           "Course Learning Objectives", lo)
            n_pages += 1
            print("[ok ] Single Page: Learning Objectives")

        weeks = sorted(p for p in tdp.glob("week_*") if p.is_dir())
        if args.max_weeks > 0:
            weeks = weeks[: args.max_weeks]
        for wk in weeks:
            wk_label = wk.name.replace("_", " ").title()  # Week 01
            sect = add_structure(rest, cid, root, wk_label, wk_label)
            # deterministic page order: overview, content_*, application, self_check, summary
            order = {"overview": 0, "application": 8, "self_check": 9, "summary": 10}
            def sort_key(p: Path):
                s = p.stem
                for k, v in order.items():
                    if s.endswith(k):
                        return (v, s)
                return (5, s)  # content_* in the middle, by name
            for html in sorted(wk.glob("*.html"), key=sort_key):
                add_singlepage(rest, cid, sect, page_title(html.name),
                               f"{wk_label} — {page_title(html.name)}", html)
                n_pages += 1
            print(f"[ok ] {wk_label}: {len(list(wk.glob('*.html')))} pages")

    return n_pages, weeks


if __name__ == "__main__":
    main()
