# The universal block-label ontology

**Status:** landed (2026-07-13) — data promoted to `schemas/taxonomies/`
**Owner directive:** "universal label ontology for a wide range of documents" (2026-07-13)
**Data files:** `schemas/taxonomies/block_kinds.json`, `schemas/taxonomies/block_relations.json`,
`schemas/taxonomies/genre_profile_*.json` (5), `schemas/taxonomies/*_lexicon.json` (the three block-ontology lexicons: `openstax_lexicon.json`, `generic_instructional_lexicon.json`, `federal_register_lexicon.json`)
**Loader:** `lib/ontology/taxonomy.py::load_taxonomy(name)` (generic `schemas/taxonomies/<name>.json` read)
**Relates to:** `SemantiK/semantik_structure/page_arranger_contract.py`, `schemas/ONTOLOGY.md`,
`lib/ontology/teaching_roles.py`, `lib/ontology/framework_blocks.py`

A three-layer label ontology so SemantiK can structure a **wide range of
documents** (textbooks, statutes, papers, forms, encyclopedias, literary prose,
infographics) with one closed structural core and open, data-driven functional
overlays.

This document merges three references: **§ A Design** (the layer model and its
governance), **§ B Source mappings** (how each inventoried gold source maps onto
the ontology), and **§ C Migration** (how the current SemantiK page-arranger
9-value enum projects onto the universal `(kind, role)` tuples).

---

# § A — Design

## The three layers

```
  L1  block_kinds       CLOSED    ~14 structural kinds     DocLayNet-compatible
       │  (a block ALWAYS has exactly one kind)
  L2  genre_profiles    OPEN(data) functional roles         attach to L1 kinds
       │  (a block has AT MOST ONE role; only when a profile resolves)
  L3  lexicons          OPEN(data) marker-string → role      publisher vocab
       (the ONLY place publisher-specific strings live)
```

- **L1 is what a block *is*** (heading, table, figure, math_block, …) — a
  document-agnostic structural fact readable from geometry + light markup, the
  same for a tax form and a physics textbook.
- **L2 is what a block *does*** in its genre (worked_example, holding, theorem,
  field_label). A role is meaningful only inside a genre; it is optional.
- **L3 is how you *recognize* an L2 role** from a specific publisher's marker
  vocabulary ("AGENCY:", the OpenStax practice markers). Pure data.

### Why closed L1 + open L2/L3 (the closed-vs-open cadence)

The structural vocabulary of documents is **small and stable** — DocLayNet
covers the world with 11 classes; we extend to 14 for code/forms/quotes. Closing
L1 buys: a fixed classifier head, DocLayNet weak-supervision, cross-eval
numbers, and a hard conservation contract (every unit gets exactly one kind).
Adding an L1 kind is a real schema change (new head class, new DocLayNet gap) —
so it is **owner-gated**.

The *functional* vocabulary is **large, genre-specific, and unbounded** — every
new corpus invents labels (a form has `checkbox_item`, a statute has `holding`).
If L2/L3 were closed, every new document genre would be a code change. So L2/L3
are **data** (`genre_profile_*.json` / `*_lexicon.json`) — onboarding a genre or
a publisher adds a JSON file, never a code edit. This is the direct realization
of the standing **wide-net rule** (owner 2026-07-04: "gates domain-agnostic;
publisher vocab = data-driven lexicon profiles, never code").

This cadence mirrors the live SemantiK lexicon system
(`lib/ontology/taxonomy.py::load_semantik_lexicon` over
`schemas/taxonomies/semantik_lexicon.json`, profile-merged, env-selected) — the
same "profiles are data" posture, generalized from the pedagogical-opener
vocabulary to the whole functional layer.

## Graceful degradation (no profile → Layer-1 only)

A document with **no matching genre profile still gets a complete labeling** —
every block resolves an L1 kind, the roles are simply absent. `nces_digest`
(statistical tables), `gutenberg` (literary prose), and `infographics`
(figure-dominant) all degrade to clean L1-only output (§ B §§ 8, 9, +). This is
a feature: the wide net catches the structural skeleton of *any* document;
profiles enrich the ones we understand. A brand-new genre ships useful (L1)
output on day one and gets richer when someone writes its profile.

Degradation is also **per-block**, not just per-document: inside an
instructional textbook, a plain body paragraph that matches no role stays
L1-`paragraph` while the worked example beside it carries the role. There is no
"unknown role" bucket — absence of a role is the honest signal.

## Multi-label policy: kind + at most one role

A block carries **exactly one L1 kind and at most one L2 role**. Rationale:

- **One kind** — the coverage/conservation invariant (every unit typed exactly
  once) is what lets SemantiK assert no content is lost; a multi-kind block
  breaks it.
- **At most one role** — roles are mutually exclusive *within a resolved
  profile* (a block is a `worked_example` **or** a `solution`, not both). If two
  markers match, the more specific / earliest-anchored wins (L3 precedence:
  source-specific lexicon before generic; longest-match before shortest). Cross-
  profile ambiguity does not arise because a document resolves ONE profile
  (§ B binds source → profile).

Relations (`schemas/taxonomies/block_relations.json`) are the escape valve for
genuinely n-ary structure: a `worked_example` block does not *contain* its
solution as a second label — it is `solution_of`-linked to a separate `solution`
block. Structure that is not a single block's identity is an EDGE, not a second
label.

## The wide-net rule (profiles/lexicons = data, never code)

Enforced mechanically:

- **No publisher-specific marker string appears outside the lexicons.**
  Verified: the OpenStax + Federal Register marker sets (`Try It`, `AGENCY:`, …)
  appear in no L1/L2 file. Generic English role labels ("guided practice") are
  role *names*, not publisher markers, and are allowed at L2.
- **A new publisher** → a new `<pub>_lexicon.json` (marker → role).
- **A new genre** → a new `genre_profile_<genre>.json` (roles + their
  `attaches_to` kinds).
- **The onboarding aligner's in-code `PRACTICE_MARKER_LEXICON` + `SOURCE_TYPE_MAPS`
  migrate to data** (§ C): the code reads the JSON, the vocab lives in the JSON.
  The class maps (`ltx_*` → kind) are structural, not publisher vocab, so they may
  stay as source-adapter code — but the *role* triggers move to lexicons.

## Governance — who may add what

| Change | Layer | Gate | Blast radius |
|--------|-------|------|--------------|
| Add/remove/rename a **kind** | L1 | **owner sign-off** (schema change) | new classifier head class, new DocLayNet gap, re-train, conservation-contract review |
| Add a **role** to a profile | L2 | maintainer (data PR) — must declare `attaches_to` ≥1 kind, map to a `teaching_role`/framework block where instructional | additive; no head change (roles are a separate, open head) |
| Add a **genre profile** | L2 | maintainer (data PR) + a § B source-mapping row | additive |
| Add a **lexicon entry / file** | L3 | maintainer (data PR) — pure vocab | additive; zero code |
| Add/rename a **relation** | relations | owner sign-off if it becomes a trained head class; else maintainer | may add a BERT-v2 class |

**Invariants any change must preserve** (mechanically checkable — see
`schemas/tests/test_block_ontology.py`):

1. Every L2 role `attaches_to` ≥1 L1 kind.
2. Every current arranger `TYPE_ENUM` value + every onboarding-aligner
   `SOURCE_TYPE_MAPS` output has a `(kind[, role])` home.
3. No publisher marker string outside the lexicons.
4. Every L1 kind declares its DocLayNet mapping (or an explicit `null` + gap
   note).
5. Every profile relation names ≥1 real L2 role.

## Authority ladder (labels earn trust)

Consistent with the BERT-v2 deployment ladder and the house `shadow → on` flag
posture: an ontology label produced by a *learned* head (vs. deterministic
markup) starts as a **hint** (a `data-*` breadcrumb), graduates to a
**validator** (proposes; deterministic code disposes) once calibrated over ≥2
corpora, and only then joins the **bulk path** — always under the fail-closed
conservation invariants. The ontology defines the label space; it does not grant
a model authority to apply it.

## Promotion history

These files were drafted in the BERT-v2 workspace and promoted into
`schemas/taxonomies/` on 2026-07-13 (owner-authorized). They are shaped for
`lib/ontology/taxonomy.py::load_taxonomy(name)` (a generic
`schemas/taxonomies/<name>.json` read). Remaining follow-up (separate, gated):
add typed loaders in `lib/ontology/` (mirroring `teaching_roles.py` /
`taxonomy.py`), and wire the onboarding aligner + the arranger contract to read
these files instead of their in-code tables (§ C).

---

# § B — Source mappings (gold sources → ontology)

For each inventoried gold source: which **Layer-2 genre profile** applies, how
its native markup maps onto **Layer-1 kinds**, and the **gaps** (what the
current onboarding-aligner `SOURCE_TYPE_MAPS` / lexicons do NOT yet resolve).

Ontology layers referenced:

- **L1** = `schemas/taxonomies/block_kinds.json` (14 closed structural kinds).
- **L2** = `schemas/taxonomies/genre_profile_*.json` (functional roles).
- **L3** = `schemas/taxonomies/*_lexicon.json` (marker → role tables).

"Currently" = the state of the BERT-v2 onboarding aligner's `SOURCE_TYPE_MAPS` +
`PRACTICE_MARKER_LEXICON` today.

| # | Source | L2 profile | Native structure signal | Status of current map |
|---|--------|-----------|------------------------|-----------------------|
| 1 | arxiv | scholarly | LaTeXML `ltx_*` classes | has class map |
| 2 | wikipedia | encyclopedic | Parsoid tag+class | has class map |
| 3 | openstax | instructional | `data-type` + `os-*` | has class + datatype + fine map |
| 4 | pmc | scholarly | JATS-derived HTML tags | generic-tag fallback only |
| 5 | cfr | legal_regulatory | eCFR HTML tags | generic-tag fallback only |
| 6 | federal_register | legal_regulatory | field-label lines | generic + **L3 lexicon** (new) |
| 7 | courtlistener | legal_regulatory | opinion HTML tags | generic-tag fallback only |
| 8 | nces_digest | *(none → L1)* | statistical tables | generic-tag fallback only |
| 9 | gutenberg | *(none → L1)* | literary prose | generic-tag fallback only |
| 10 | forms | forms | field/label layout | generic-tag fallback only |
| + | mkdocs-site | instructional | clean mkdocs HTML | (import-docs path) |
| + | infographics | *(none → L1)* | figure-dominant | (figure path) |

---

## 1. arxiv — profile: `scholarly`

**L1 mapping (from the arxiv class map):** `ltx_title*` → `heading` (document
title → `title_block`); `ltx_p`/`ltx_para` → `paragraph`; `ltx_caption` →
`caption`; `ltx_table`/`ltx_tabular` → `table`; `ltx_itemize`/`ltx_enumerate`/
`ltx_item` → `list_item`.

**L2 roles:** `ltx_theorem*` → `(paragraph|math_block, theorem)`; `ltx_proof`
→ `(paragraph, proof)`. `abstract`, `lemma`, `reference_entry`,
`author_block`, `affiliation` are declared in the scholarly profile but not yet
mapped from `ltx_*`.

**Gaps:** (a) `ltx_equation`/`ltx_equationgroup` currently coerce to
`paragraph` — should be **L1 `math_block`** (the whole point of the formula
kind + the `SEMANTIK_MATH_RECONSTRUCT` MathML path). (b) `ltx_theorem` currently
coerces to `definition_box` — under the new ontology it is L1 `paragraph` + L2
role `theorem` (scholarly), not the instructional `definition_box`. (c) abstract
/ author / affiliation / bibliography containers unmapped.

## 2. wikipedia — profile: `encyclopedic`

**L1 mapping (from the wikipedia class map):** `infobox`/`wikitable` → `table`;
`thumbcaption`/`gallerytext` → `caption`; body via generic-tag fallback
(`p`→paragraph, `hN`→heading, `li`→list_item, …).

**L2 roles:** `infobox_row` (on `table`/`list_item`), `hatnote`,
`see_also_entry`.

**Gaps:** `hatnote` + `see_also_entry` have no class/tag detector yet (they are
plain `<p>`/`<li>` in Parsoid, distinguished only by position/leading-text — a
future L3 encyclopedic lexicon or a structural heuristic). `infobox_row` is not
emitted (the infobox is captured whole as one `table`).

## 3. openstax — profile: `instructional`

**L1 mapping (from the openstax datatype + class maps):**
`title`/`document-title` → `heading`; `os-caption` → `caption`; `equation` →
(currently `paragraph`; should be `math_block` — same gap as arxiv).

**L2 roles:** `example` → `worked_example`; `solution`/`solution-title` →
`solution`; `note`/`term`/`os-note-body` → `definition_box`;
`problem`/`exercise`/`os-problem-container` → `exercise_item`; `try`/os fine →
`guided_practice`. **L3:** `openstax_lexicon.json` triggers
`guided_practice` / `worked_example` / `exercise_item` from the OpenStax marker
strings (confined to that lexicon).

**Gaps:** `answer_item`, `learning_objectives`, `summary`, `review`,
`key_terms` roles are declared but resolved today only if a heading/lexicon
marker names them — no `data-type` for them in the class map. The `equation` →
`math_block` fix applies here too.

## 4. pmc — profile: `scholarly`

**L1 mapping:** generic-tag fallback only (`p`/`hN`/`table`/`figcaption`/`li`).

**L2 roles:** none resolved yet — PMC's JATS→HTML keeps section semantics in
`sec-type`/`article-*` attributes the current map ignores.

**Gaps:** no PMC class map. `abstract` (JATS `<abstract>`), `reference_entry`
(`<ref>`), `author_block`/`affiliation` (`<contrib>`) are all reachable from
JATS-derived attributes — a PMC class map (or an `sec-type` datatype map) is
the onboarding step. Reuse the scholarly profile's roles.

## 5. cfr — profile: `legal_regulatory`

**L1 mapping:** generic-tag fallback (`p`→paragraph, `hN`→heading, `table`).

**L2 roles:** `section_number` (the `§ N.M` heading), `citation`,
`enacting_clause` — resolved today only via heading text, not markup.

**Gaps:** no CFR class map; eCFR HTML carries `data-*`/`class` hooks
(section wrappers, authority notes) that a CFR class map could bind to
`section_number` / `citation`. A future L3 CFR lexicon can catch leading
`Authority:` / `Source:` label lines the way the Federal Register lexicon does.

## 6. federal_register — profile: `legal_regulatory`

**L1 mapping:** generic-tag fallback for body.

**L2 roles + L3 lexicon:** `federal_register_lexicon.json` maps the
leading field labels → roles: `agency_header`, `docket_line` (action/dates),
`enacting_clause` (summary). This is the seed L3 lexicon for the legal profile.

**Gaps:** the label lexicon covers the front-matter block; the numbered
regulatory body (amendatory instructions, section-by-section) still degrades to
`paragraph`/`heading` with no `section_number`/`citation` role. A body-level
CFR-style map (shared with source 5) would extend coverage.

## 7. courtlistener — profile: `legal_regulatory`

**L1 mapping:** generic-tag fallback.

**L2 roles:** `holding`, `syllabus`, `citation`.

**Gaps:** no class map and no reliable markup signal — court opinions are
mostly undifferentiated `<p>`. `holding`/`syllabus`/`citation` need a
structural/positional heuristic or an L3 lexicon (e.g. a leading `Syllabus`
/ `Held:` marker), the same pattern as the Federal Register lexicon.

## 8. nces_digest — profile: **none → Layer-1 only** (graceful degradation)

**L1 mapping:** table-dominant statistical digest — `table` (the load-bearing
kind), `heading`, `paragraph`, `caption`, `furniture` (running page furniture).

**L2 roles:** none. No genre profile fits a statistical digest, so it degrades
cleanly to Layer-1 (the whole point of the graceful-degradation rule — a
document with no profile still gets a complete, DocLayNet-compatible structural
labeling).

**Gaps:** if statistical-table semantics (table title vs. source-note vs.
footnote) become worth modeling, that is a *new* profile (`statistical`), not a
change to L1. Until then, `table` + `caption` + `footnote` carry it.

## 9. gutenberg — profile: **none → Layer-1 only** (graceful degradation)

**L1 mapping:** literary prose — `title_block`, `heading` (chapter headings),
`paragraph`, `blockquote` (verse / quoted passages), `separator` (thematic
breaks), `furniture` (Project Gutenberg boilerplate header/footer).

**L2 roles:** none — no pedagogical/legal/scholarly structure.

**Gaps:** Gutenberg's license boilerplate is best caught as `furniture`
(candidate for a cross-course boilerplate-dedup pass). A future `literary`
profile (chapter / verse / stage-direction roles) is possible but out of scope.

## 10. forms — profile: `forms`

**L1 mapping:** the natural host is L1 `form_field`; layout text is
`paragraph`/`heading`; instructions are `list_item`/`paragraph`.

**L2 roles:** `field_label`, `field_value`, `instruction`, `checkbox_item`.

**Gaps:** the onboarding aligner has no `forms` class map — it uses the
generic-tag fallback, so `form_field` is never minted (fields arrive as
`paragraph`). Form detection needs a geometry/layout signal (label-value
adjacency, checkbox glyphs) that the HTML-truth aligner cannot see; this is a
**VLM/geometry** onboarding step, and `form_field` is the one L1 kind with **no
DocLayNet parent** (so DocLayNet weak-supervision can't bootstrap it).

## + mkdocs-site — profile: `instructional`

**L1 mapping:** clean mkdocs HTML (via `ed4all import-docs`) → `heading`,
`paragraph`, `list_item`, `code_block` (heavy — tutorial docs), `table`,
`figure`/`caption`.

**L2 roles:** instructional roles via headings + `generic_instructional_lexicon.json`
(guided_practice markers). `summary` / `key_terms` from section headings.

**Gaps:** `code_block` is prominent here and folds to DocLayNet `text` (documented
L1↔DocLayNet gap — see below). Otherwise the cleanest source: no OCR, real HTML
structure, so alignment is near-lossless.

## + infographics — profile: **none → Layer-1 only** (graceful degradation)

**L1 mapping:** figure-dominant — `figure` (the payload), `caption`, `heading`,
short `paragraph`, `furniture`.

**L2 roles:** none.

**Gaps:** the semantic content lives *inside* the image, so **VLM alt-text /
extended description** (`SEMANTIK_FIGURE_CAPTION`) is load-bearing — the block
ontology only labels the figure envelope, not the infographic's internal
structure. A DocLayNet `picture` label bootstraps the envelope detection.

---

# § C — Migration (arranger 9-kind enum → universal tuples)

How the current SemantiK page-arranger contract (`TYPE_ENUM`, 9 values) and the
BERT-v2 dataset migrate onto the 3-layer universal ontology **without breaking
byte-compatibility** for existing arranger output.

## 1. Arranger `TYPE_ENUM` → (Layer-1 kind, Layer-2 role)

`SemantiK/semantik_structure/page_arranger_contract.py::TYPE_ENUM` conflates
structural KIND and pedagogical ROLE into one flat 9-value enum. The universal
ontology splits them: every arranger value has a home as a **kind** plus an
optional **role**. This is byte-compatible on READ — the projection below is a
pure relabeling, no arrangement is invalidated.

| arranger `TYPE_ENUM` | L1 kind | L2 role (profile) | Notes |
|----------------------|---------|-------------------|-------|
| `heading` | `heading` | — | `level` attr preserved; document title → `title_block` |
| `paragraph` | `paragraph` | — | the fallback kind |
| `table` | `table` | — | 1:1 |
| `figure_caption` | `caption` | — | (splits: a real image becomes L1 `figure` + `caption`) |
| `example` | `paragraph` | `worked_example` (instructional) | kind is prose; role names the pedagogy |
| `solution` | `paragraph` | `solution` (instructional) | " |
| `exercise_list` | `list_item` | `exercise_item` (instructional) | list granularity is per-item in L1 |
| `definition_box` | `paragraph` | `definition_box` (instructional) | also valid on `blockquote` |
| `furniture` | `furniture` | — | `subkind` attr (running_header/page_number/…) |

**Byte-compat rule:** an arranger consumer that only knows the 9-value enum can
keep reading the flat value; a `(kind, role)`-aware consumer reads the tuple.
The projection is total (every enum value maps) and lossless for the 4 pure-kind
values; the 4 pedagogical values GAIN a role but their kind is unambiguous. The
2 relabels that change the surface value are `figure_caption`→`caption` (a rename
+ the `figure` split) and the pedagogical values acquiring a role — both are
additive to a v3 contract, see § 4.

## 2. Onboarding-aligner `SOURCE_TYPE_MAPS` changes

Today the maps target the flat 9-value `TYPE_ENUM`. Under the universal
ontology they target `(kind, role)`:

- **arxiv class map:** `ltx_equation`/`ltx_equationgroup` → **`math_block`**
  (was `paragraph`); `ltx_theorem*` → `(paragraph, theorem[scholarly])`
  (was `definition_box`); `ltx_proof` → `(paragraph, proof)`; keep
  `ltx_caption` → `caption`, `ltx_table` → `table`, `ltx_item*` → `list_item`.
- **openstax datatype map:** `example` → `(paragraph, worked_example)`;
  `solution*` → `(paragraph, solution)`; `note`/`term` →
  `(paragraph, definition_box)`; `problem`/`exercise` →
  `(list_item, exercise_item)`; `equation` → `math_block` (was `paragraph`).
  The fine-class `try → guided_practice` becomes a first-class role, not a
  side-channel.
- **wikipedia class map:** `infobox`/`wikitable` → `(table, infobox_row?)`;
  captions → `caption`.
- **New maps** for pmc / cfr / courtlistener / forms (today generic-tag
  fallback) as described in § B.
- **`PRACTICE_MARKER_LEXICON`** (the in-file dict) **moves out of code** into
  `openstax_lexicon.json` + `generic_instructional_lexicon.json`. The aligner's
  practice-marker fine-typer reads the lexicon files instead of the hardcoded
  dict — enforcing the wide-net rule (publisher vocab = data).

**Generic-tag fallback is unchanged** — it already emits L1 kinds
(`hN`→heading, `table`→table, `figcaption`→caption, `dt`/`dd`→
definition_box→now `(paragraph, definition_box)`, `li`→list_item/paragraph).

## 3. BERT-v2 dataset class mapping

The BERT-v2 heads train on arranger label-factory records. Their class spaces
map to the universal ontology's relations
(`schemas/taxonomies/block_relations.json`) and kinds:

**RELATION head** (currently binary `none|same_unit`; 5-class target):

| BERT-v2 class | universal relation | family |
|---------------|--------------------|--------|
| `none` | (no edge) | — |
| `same_unit` | `same_unit` | structural |
| `caption_of` | `caption_of` | structural |
| `solution_of` | `solution_of` | profile (needs `solution`/`worked_example` roles) |
| `continues` | `continues` | structural |

**Co-occurrence classes** (kept as training signal; from the co-occurrence
deriver):

| co-occurrence edge | universal relation | family |
|--------------------|--------------------|--------|
| `adjacency` | `adjacent` | structural |
| `section_comembership` | `same_section` | structural |
| `practice_of` | `practice_of` | profile (needs `guided_practice`/`worked_example`) |

**FURNITURE head** (binary `content|furniture`): maps directly to L1
`furniture` (a KIND, not a relation) — a block's kind IS the label. The
supply-blocker (the arranger *drops* furniture instead of labeling it) is fixed
by the L1 rule that furniture is **listed, never dropped** — see § 4.

**Provenance note:** the relation head's minority classes (`caption_of`,
`solution_of`, `continues`) are supply-blocked today (5-class is unevaluable on
the v2 corpus). The universal `practice_of` / `same_section` / `adjacent`
classes are *dense* (derived from every page), so they are the trainable
co-occurrence signal even before the arranger supplies more minority relations.

## 4. Arranger contract v3 (what it would emit)

A v3 arrangement block would carry the tuple explicitly instead of the flat
enum:

```json
{"ids": ["p3_u12", "p3_u13"],
 "kind": "paragraph",
 "role": "worked_example",
 "profile": "instructional",
 "level": null,
 "subkind": null,
 "continues_prev_page": false}
```

Changes from v2 (`CONTRACT_VERSION = 2`):

1. **`type` → `kind` + `role` + `profile`.** `kind` is a closed L1 value;
   `role`/`profile` are optional (absent ⇒ Layer-1-only, the graceful-degrade
   path). A v2 `type` value maps in via the § 1 table (accepted on read).
2. **Furniture is LABELED, never dropped.** The v2 contract lets the teacher
   omit furniture units; v3 REQUIRES every furniture unit in a `furniture`
   block with a `subkind` — this is exactly the fix the BERT-v2 FURNITURE head
   is supply-blocked on (16/894 positives because the teacher drops them). The
   coverage invariant (every unit id exactly once) already enforces "no dropped
   id"; v3 makes furniture a first-class LABELED kind.
3. **`figure_caption` splits** into L1 `figure` (the image) + `caption` (its
   text), bound by `caption_of` — v2 collapsed both into one value.
4. **`math_block` / `code_block` / `blockquote` / `footnote` / `form_field` /
   `title_block` / `separator`** become expressible (v2's 9 values could not
   name display math, code, or form fields — they all fell to `paragraph`).
5. **`CONTRACT_VERSION` bump 2 → 3.** The v2 alias table (`TYPE_ALIASES`) is
   preserved as the read-compat shim: `text→paragraph`, `worked_example→
   (paragraph, worked_example)`, `section→heading`, etc.

## 5. DocLayNet-compat note (public-dataset leverage)

Layer-1 is DocLayNet-v1-compatible **by construction**
(`schemas/taxonomies/block_kinds.json::x-doclaynet`), which buys three things:

- **Pre-training / weak supervision.** DocLayNet (~80k human-labeled pages, 11
  classes, CDLA-Permissive-ish) can be ingested directly as L1 labels — its 11
  classes each have an L1 home, so a DocLayNet page is a labeled example for the
  structural head with zero re-annotation.
- **Cross-eval.** A SemantiK L1 labeling can be scored against DocLayNet ground
  truth by collapsing to the 11-class projection (`x-doclaynet` map), giving a
  public benchmark number.
- **Honest gaps.** 5 L1 kinds have **no DocLayNet parent**: `code_block`,
  `blockquote` (both fold to DocLayNet `text` — lossy but ingestible),
  `form_field`, `separator`, and `furniture[watermark]` (no DocLayNet class at
  all). DocLayNet supervision therefore bootstraps 9 of 14 kinds; the other 5
  need SemantiK's own labels (or geometry, for `form_field`). This is the
  documented limit of the public-dataset leverage — it is a head start, not a
  complete teacher.
