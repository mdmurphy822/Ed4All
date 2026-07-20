# Ed4All × OpenOLAT Retrieval-Demo — Runbook

Bring up an OpenOLAT course that embeds the Ed4All **grounded-ask widget**, with
the `alg-glm-02` course serving live answers. Two independent Docker stacks:

| Stack | Compose file | Serves | Port(s) |
|---|---|---|---|
| **LMS shell** | `demo/openolat/docker-compose.yml` | OpenOLAT + Postgres | `8080` |
| **Ed4All GUI** | `docker-compose.yml` (repo root) | learner SPA + ask API + ollama | `8077` (+ `11435` loopback) |

The widget is an OpenOLAT *External Page* element that iframes
`http://localhost:8077/learn/?course=alg-glm-02`. The browser loads both origins
directly, so they never need to reach each other server-to-server.

Architecture, suitability verdict, and import fidelity: **`FIDELITY.md`**.

---

## Prerequisites

- Docker Engine with Compose v2 (`docker compose`). Verified on aarch64 (GB10).
- The course build present at `LibV2/courses/alg-glm-02/` with a
  `vector_index/` (ships in the repo checkout). Check:
  `curl -s http://127.0.0.1:8077/api/learn/ask-ready/alg-glm-02` →
  `{"exists":true,"has_vector_index":true}` (after the GUI stack is up).
- Answer model in the ollama sidecar (`qwen2.5:7b-instruct-q4_K_M`). If the
  `ed4all_ollama-models` volume is fresh, pull it once (see step 2).
- Optional GPU: the root `docker-compose.override.yml` reserves the NVIDIA GPU
  for the ollama sidecar. CPU-only works (slower first answer).

No secrets live in any tracked file. The OpenOLAT admin login is the **product
default** `administrator` / `openolat`, auto-created on first boot — treat it as
a demo credential and change it (step 5) before any network exposure.

---

## 1. Start the Ed4All GUI stack (widget backend)

From the repo root:

```bash
docker compose up -d                 # gui + ollama (studio mode)
curl -s http://127.0.0.1:8077/api/health                       # → 200
curl -s http://127.0.0.1:8077/api/learn/ask-ready/alg-glm-02   # → has_vector_index:true
```

## 2. (First run only) pull the answer model into the sidecar

```bash
docker compose exec ollama ollama pull qwen2.5:7b-instruct-q4_K_M
docker compose exec ollama ollama list        # confirm it's present
```

Smoke-test a grounded answer:

```bash
curl -s -X POST http://127.0.0.1:8077/api/learn/ask \
  -H 'Content-Type: application/json' \
  -d '{"slug":"alg-glm-02","query":"what is the greatest common factor?"}' \
  | python3 -c 'import sys,json;a=json.load(sys.stdin)["answer"];print(a["status"],a["engine"],len(a["citations"]),"citations")'
# → answered hybrid-rrf 5 citations
```

## 3. Start the OpenOLAT stack

```bash
cd demo/openolat
docker compose up -d --build          # builds ed4all-openolat:19.1.6 (native arm64)
docker compose ps                     # wait for both healthy (first boot ~1-3 min)
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/openolat/dmz/   # → 200
```

First boot auto-creates the Postgres schema (~321 tables) and the default admin
user. The image build pulls `frentix/openolat:19.1.6` once as a *file source*
only — see `FIDELITY.md` § "arm64 deploy".

## 4. Provision the demo course (one command)

The course slug is machine-local demo data, so it is **not** baked into any
tracked file — supply it via a gitignored `.env` (or `--slug`):

```bash
cd demo/openolat
cp demo.env.example .env        # first time: set ED4ALL_DEMO_SLUG=<your course>
set -a; . ./.env; set +a        # load ED4ALL_DEMO_SLUG / ED4ALL_WIDGET_BASE
python3 provision.py            # one command
# …or without the .env:
#   python3 provision.py --slug <your-course-slug> --widget-base http://localhost:8077
```

Creates (idempotent — re-run reuses an existing course and just ensures the
learner; use `--force` to rebuild from scratch):

- a published course `alg-glm-02 — Elementary Algebra (Ed4All retrieval demo)`;
- 10 week Structure nodes + 93 Single Page reading nodes (the cartridge HTML);
- the **External Page** widget element → the grounded-ask surface;
- a demo learner `demo-learner`.

The script prints the **Course run URL**, admin console, widget URL, and learner
login on success. Credentials are passed as flags/env, never written to disk:

| Purpose | Flag | Env | Demo default |
|---|---|---|---|
| Admin user / pass | `--admin-user` / `--admin-pass` | `OLAT_ADMIN_USER` / `OLAT_ADMIN_PASS` | `administrator` / `openolat` |
| Learner pass / email | `--learner-pass` / `--learner-email` | `OLAT_LEARNER_PASS` / `OLAT_LEARNER_EMAIL` | `demo-learner-pw` / `demo-learner@example.com` |
| Widget base URL | `--widget-base` | `ED4ALL_WIDGET_BASE` | `http://localhost:8077` |

## 5. Change the admin password (before any exposure)

```
Open http://localhost:8080/openolat/  → log in administrator / openolat
→ Profile ▸ Password  → set a strong password.
```

Then re-provision with `--admin-pass <new>` (or export `OLAT_ADMIN_PASS`).

## 6. See the demo

- **Course + widget:** open the *Course run URL* printed by `provision.py`
  (`http://localhost:8080/openolat/url/RepositoryEntry/<key>`), log in as the
  learner (or admin), open the **"Ask the Course"** element → the embedded
  Ed4All ask surface. Ask *"what is the greatest common factor?"* → grounded
  answer with citations.
- **Reading content:** the *Week NN* modules hold the imported HTML pages.

> The `?course=` auto-pin (hide the picker) and the frame-ancestors lock are
> TRACK-EMBED items E1/E2. Until E1 lands, pick `alg-glm-02` in the framed
> widget's course selector. To lock framing to OpenOLAT once E2 lands, set
> `ED4ALL_GUI_FRAME_ANCESTORS=http://localhost:8080` in the root stack env.

---

## Stop / teardown

```bash
# Stop, keep data (course, DB, models persist on named volumes):
cd demo/openolat && docker compose stop
cd ../..          && docker compose stop

# Full teardown INCLUDING data (drops the OpenOLAT DB + course):
cd demo/openolat && docker compose down -v
# (the root stack's ed4all-data / ollama-models volumes are left intact unless
#  you also run `docker compose down -v` at the repo root)
```

## Host-specific overrides (gitignored)

Copy `docker-compose.override.yml.example` → `docker-compose.override.yml` to
remap ports, enlarge the JVM heap, or change the DB password (mirror any DB
password change into `config/olat.local.properties` **and**
`config/openolat.xml`, then rebuild). The override file is gitignored.

## Troubleshooting

- **OpenOLAT slow to answer on first boot:** schema creation + Spring init take
  1-3 min; `docker compose logs -f openolat` and wait for
  `Server startup in [NNNNN] milliseconds`.
- **Widget shows a course picker instead of alg-glm-02:** expected until
  TRACK-EMBED E1; select the course manually.
- **Ask returns `refused_low_confidence`:** the question is out of the corpus —
  the retrieval floor working as designed. Try a course-topic question.
- **Port 8080 taken:** set an alternate mapping in the gitignored override.
