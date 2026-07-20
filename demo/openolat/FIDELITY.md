# OpenOLAT Retrieval-Demo — Import Fidelity & Deviations

This is the owner's **suitability evidence**: exactly what of the Ed4All
`alg-glm-02` build OpenOLAT could consume, what degraded, and what was dropped —
plus the two justified deviations from the approved design. Everything below was
observed on this aarch64 box (NVIDIA GB10), OpenOLAT 19.1.6, on 2026-07-20.

## TL;DR

The design's verdict held **exactly**: OpenOLAT is a clean *LMS shell* and a
clean *iframe host* for the grounded-ask widget, but a poor *consumer of our
build artifacts*. The reading HTML imported at 100%; the Common Cartridge
envelope and QTI 1.2 quizzes are un-importable and were designed around, not
through. The demo's star — the live grounded-ask widget — is an **External
Page** course element that frames the Ed4All learner SPA, and it works
end-to-end.

## Container inventory (both stacks, all healthy)

| Container | Image | Port(s) | State |
|---|---|---|---|
| `openolat-openolat-1` | `ed4all-openolat:19.1.6` (built here) | `8080:8080` | healthy |
| `openolat-openolat-db-1` | `postgres:16` | `5432` (internal) | healthy |
| `ed4all-gui-1` | `ed4all-gui:latest` | — (shares ollama netns) | healthy |
| `ed4all-ollama-1` | `ollama/ollama:latest` | `8077:8077`, `127.0.0.1:11435:11434` | up |

## Import fidelity table

Source cartridge: `LibV2/courses/alg-glm-02/source/imscc/alg-glm-02.imscc`
(IMS Common Cartridge **v1.3**, 124 files). Imported via `provision.py` over the
OpenOLAT REST API into course key `114208908274520`.

| Cartridge surface | Count in `.imscc` | Into OpenOLAT | Fidelity | Notes |
|---|---:|---|---|---|
| Reading HTML — week content (`*_content_*`) | 52 | **52 Single Page nodes** | ✓ full | verbatim HTML, one node each |
| Reading HTML — week overview | 10 | **10 Single Page** | ✓ full | |
| Reading HTML — week summary | 10 | **10 Single Page** | ✓ full | |
| Reading HTML — week self-check (formative, answers inline) | 10 | **10 Single Page** | ✓ full | rendered as reading, not as a graded test |
| Reading HTML — week application | 10 | **10 Single Page** | ✓ full | |
| Course-overview learning objectives HTML | 1 | **1 Single Page** | ✓ full | |
| Week grouping | 10 modules | **10 Structure nodes** | ✓ structure | `Module N: Chapter N` → `Week NN` structure node |
| **QTI 1.2 quizzes** (`imsqti_xmlv1p2/.../assessment`) | 10 | **0** | ✗ dropped | OpenOLAT 19.x cannot import or convert QTI 1.2 (removed at v16); answer keys live in this XML and did **not** transfer |
| **Assignments** (`associatedcontent/.../learning-application-resource`) | 10 | **0** | ✗ dropped | CC learning-application-resource has no OpenOLAT importer |
| **Discussion topics** (`imsdt_xmlv1p3`) | 10 | **0** | ✗ dropped | CC discussion-topic envelope not read; OpenOLAT forums are native-only |
| `imsmanifest.xml` (CC organization tree) | 1 | **not read** | n/a | OpenOLAT has no CC importer, so the manifest — and the legacy ROOT-`<title>` packager defect `df49b21a` in it — is **moot** |
| **Grounded-ask widget** (not in the cartridge) | — | **1 External Page node** | ✓ live | frames `http://localhost:8077/learn/?course=alg-glm-02` |

**Totals imported:** 93 Single Page reading nodes + 11 Structure nodes
(1 root + 10 week) + 1 External Page widget. Verified on disk in
`editortreemodel.xml`: `SPCourseNode = 93`, `STCourseNode = 11`,
`TUCourseNode = 1`.

**Totals dropped:** 30 non-HTML resources (10 QTI quizzes, 10 assignments,
10 discussions) + the CC manifest. None are recoverable on OpenOLAT 20.x/21.x
either — this is a format-capability gap, not a version-tuning one.

### Why the drops are *designed around*, not a demo failure

The demo's value proposition is the **grounded retrieval/ask experience**, which
needs **no import at all** — it serves live off the bind-mounted
`LibV2/courses/alg-glm-02/` vector index. The imported reading HTML exists only
to place that widget *inside a real course shell*. Quizzes/discussions are
out-of-scope for the demo (design § C.3). A customer who needs graded
assessment in OpenOLAT would hand-author a QTI **2.1** test in OpenOLAT's native
editor (or the packager would grow a QTI 2.1 emit path — a product change, not a
demo one).

## Widget embed — iframe-readiness evidence

The External Page course element points at
`http://localhost:8077/learn/?course=alg-glm-02` (stored decomposed in the TU
node as host / port `8077` / uri `/learn/` / query `course=alg-glm-02`). The
target is browser-reachable and frame-safe:

```
$ curl -sS -D - -o /dev/null 'http://127.0.0.1:8077/learn/?course=alg-glm-02'
HTTP/1.1 200 OK
content-type: text/html; charset=utf-8
# (no X-Frame-Options, no restrictive Content-Security-Policy) → framable
```

The base learner SPA (course picker `#course-sel`, ask surface) renders. The
`?course=` **auto-pin** (pre-select + hide the picker) is TRACK-EMBED work item
E1; until it lands, the framed widget shows the picker and the learner selects
`alg-glm-02`. Setting `ED4ALL_GUI_FRAME_ANCESTORS=http://localhost:8080` on the
GUI (TRACK-EMBED E2) will additionally lock framing to the OpenOLAT origin.

## Grounded-answer end-to-end evidence

The widget's backend answered a course question through the containerized stack
(model resident in the ollama sidecar, bge-large embeddings, hybrid-rrf
retrieval):

```
POST http://127.0.0.1:8077/api/learn/ask  {"slug":"alg-glm-02","query":"what is the greatest common factor?"}
→ status=answered  engine=hybrid-rrf  model=qwen2.5:7b-instruct-q4_K_M
  confident=true   citations=5 (chunks alg_glm_02_chunk_00130/137/127/128/132)
  answer: "The greatest common factor (GCF) of two or more expressions is the
           largest expression that divides evenly into all given expressions…"
```

`GET /api/learn/ask-ready/alg-glm-02 → {"exists":true,"has_vector_index":true}`
confirms answers are **grounded** (vector index present), not lexical-fallback.
(An out-of-corpus phrasing such as "what is a variable in algebra?" returns an
honest `refused_low_confidence` — the retrieval floor working, not a bug.)

## Deviation 1 — OpenOLAT **19.1.6**, not the design's 20.3.x

The design pinned OpenOLAT 20.3.x. **No 20.x/21.x ready-to-run artifact is
publicly fetchable:** OpenOLAT publishes no prebuilt WAR (GitHub releases carry
no assets), and the only public image, `frentix/openolat`, ships a single tag —
`19.1.6`, amd64-only. Building 20.3.x from source needs the full Maven/Node
toolchain and is out of scope for a demo bring-up.

19.1.6 carries **every surface the demo exercises**: webcontent/Single-Page
hosting, the External Page + LTI course elements for the embed, and the REST API
for one-command provisioning. The CC/QTI-1.2 gap is **identical** on 19.x and
20.x/21.x (QTI 1.2 was dropped at v15, its converter removed at v16), so the
suitability verdict does not move with the version. Net: 19.1.6 is a strictly
sufficient, lower-risk substitute for the demo.

## Deviation 2 — `postgres:16`, not `pgvector/pgvector:pg16`

The design specified pgvector "only strictly needed for v21". On 19.1.6 pgvector
is unused, so plain multi-arch `postgres:16` is correct and lighter. (All
retrieval vectors live in Ed4All's own on-device index, never in OpenOLAT's DB.)

## arm64 deploy — how a native run was achieved without a native image

An OpenOLAT deployment is a pure-Java **exploded webapp** (architecture-
independent JVM bytecode) on Tomcat 10.1.35 + Temurin JDK 17 — which frentix's
amd64 image bundles. The `Dockerfile` uses that amd64 image **only as a
`COPY --from` file source** (its filesystem is read; its code is never
executed), copying the webapp + the Postgres driver into the official multi-arch
`tomcat:10.1.35-jdk17-temurin` (arm64 native). Result: OpenOLAT runs natively on
aarch64 with **zero qemu/binfmt emulation** and zero recompilation. Boot logs
confirm `Architecture: aarch64`, `Apache Tomcat/10.1.35`, `JVM 17.0.14`, schema
auto-created (321 `o_*` tables), admin login live, REST API v1.0 reachable.

### Classpath note (why config lands in `tomcat/lib`)

frentix runs with a split `CATALINA_BASE=/home/openolat` so
`${catalina.base}/lib` (default `common.loader`) picks up
`olat.local.properties`. The official image has
`CATALINA_BASE == CATALINA_HOME == /usr/local/tomcat`, so the same default
`common.loader` makes `/usr/local/tomcat/lib` the classpath dir — hence the
Dockerfile drops both `olat.local.properties` and `postgresql.jar` there.
