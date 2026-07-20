# The universal block-label ontology

**What it is:** a three-layer label vocabulary for structuring documents of many genres, with one
closed structural core and open, data-driven functional overlays.
**Where it lives:** entirely in **data files** under `schemas/taxonomies/` — there is no Python module
that owns this vocabulary. The only code is a generic loader and a test suite.
**Loader:** `lib/ontology/taxonomy.py::load_taxonomy(name)` — a generic cached
`schemas/taxonomies/<name>.json` read.
**Invariant tests:** `schemas/tests/test_block_ontology.py`.
**Relates to:** `docs/architecture/hybrid-vision-extraction.md`, `schemas/ONTOLOGY.md`,
`lib/ontology/teaching_roles.py`, `lib/ontology/framework_blocks.py`.

> **Read § Adoption status before treating any of this as live behavior.** The data is real, verified,
> and test-enforced. Its consumption by the shipping converter is partial and precisely bounded.

---

## 1. The three layers

```
  L1  block_kinds       CLOSED     16 structural kinds     DocLayNet-mapped
       │  (a block ALWAYS has exactly one kind)
  L2  genre_profiles    OPEN(data) functional roles        attach to L1 kinds
       │  (a block has AT MOST ONE role; only when a profile resolves)
  L3  lexicons          OPEN(data) marker string → role    publisher vocabulary
       (the ONLY place publisher-specific strings live)
```

- **L1 is what a block *is*** (heading, table, figure, math_block, …) — a document-agnostic structural
  fact readable from geometry plus light markup, identical for a tax form and a physics textbook.
- **L2 is what a block *does*** in its genre (worked_example, holding, theorem, field_label). A role is
  meaningful only inside a genre, and is optional.
- **L3 is how you *recognize*** an L2 role from a specific publisher's marker vocabulary. Pure data.

### Why closed L1, open L2/L3

The structural vocabulary of documents is small and stable — DocLayNet covers the world with 11
classes; L1 extends to 16. Closing L1 buys a fixed classifier head, DocLayNet weak supervision,
cross-eval numbers, and a hard conservation contract (every unit gets exactly one kind). Adding an L1
kind is a real schema change and is maintainer-gated.

The *functional* vocabulary is large, genre-specific, and unbounded — a form invents `checkbox_item`, a
statute invents `holding`. If L2/L3 were closed, every new genre would be a code change. So they are
data: onboarding a genre or a publisher adds a JSON file, never a code edit.

This is the direct realization of the standing **wide-net rule**: structural gates stay domain-agnostic,
and publisher-specific vocabulary lives in data-driven lexicons rather than in code. A publisher marker
compiled into a Python module is a corpus-specific gate wearing a general-purpose disguise — it passes
on the corpus it was written against and silently under-labels every other one.

The cadence mirrors the already-live SemantiK lexicon system
(`lib/ontology/taxonomy.py::load_semantik_lexicon` over `schemas/taxonomies/semantik_lexicon.json`,
profile-merged, env-selected), generalized from pedagogical-opener vocabulary to the whole functional
layer.

---

## 2. The data, as it exists on disk

### L1 — `schemas/taxonomies/block_kinds.json`

`version: 0.2.0-draft`, `layer: 1`, `closed: true`. The `$defs.BlockKind.enum` has **16** values, and
`x-block-kinds` carries one descriptor entry per kind:

```
heading   paragraph   list_item   table   figure   chart   caption   math_block
code_block   blockquote   aside   footnote   form_field   title_block   separator   furniture
```

Each `x-block-kinds` entry carries `kind`, `label`, `description`, `attributes[]`, `subkinds[]`, a
`doclaynet` key (the DocLayNet class it folds onto, or `null`), and a `doclaynet_note` explaining the
fold. A top-level `x-doclaynet-coverage` block carries the inverse `map` (all 11 DocLayNet classes →
an L1 home) plus an explicit `ours_without_doclaynet_parent` list.

`chart` is deliberately distinct from `figure`: its accessibility contract is a structured **data
description**, not merely alt text, so a graph's underlying data survives for a screen-reader user
rather than collapsing to a one-line label. `aside` renders as `role=complementary` and is excluded from
the main reading sequence. Both fold lossily onto DocLayNet (`aside`→`text`, `chart`→`picture`) since
DocLayNet has no native parent for either.

**DocLayNet-compat leverage and its documented limit.** Two distinct facts are easy to conflate here,
so state them separately.

- **Only two kinds declare `doclaynet: null`** — `form_field` and `separator`. The other 14 name a
  DocLayNet class.
- **`x-doclaynet-coverage.ours_without_doclaynet_parent` lists seven entries** — `code_block`,
  `blockquote`, `aside`, `chart`, `form_field`, `separator`, and `furniture[subkind=watermark]`. That
  list is broader than the `null` count because it also names kinds that *do* declare a parent but
  share it **many-to-one**: `code_block` / `blockquote` / `aside` all fold onto `text` alongside
  `paragraph`, and `chart` folds onto `picture` alongside `figure`. The last entry is a *subkind*, not
  a kind — `furniture` itself maps to `page-header | page-footer`.

The consequence for weak supervision: a DocLayNet label can never *discriminate* those five from the
sibling kind they share a parent with, and supplies nothing at all for the two `null` kinds. It is a
head start, not a complete teacher.

### L2 — `schemas/taxonomies/genre_profile_*.json` (8 files)

Each declares `layer`, `closed`, a `profile` object (`id` + `label`), and an `x-roles` array:

| Profile id | Roles |
|---|---:|
| `instructional` | 10 |
| `legal_regulatory` | 7 |
| `literary` | 7 |
| `scholarly` | 7 |
| `forms` | 4 |
| `statistical` | 5 |
| `technical_manual` | 5 |
| `encyclopedic` | 3 |

An `x-roles` entry carries `role`, `label`, `description`, `attaches_to[]` (the L1 kinds it may sit on),
and — where instructional — `maps_to_teaching_role` and `framework_block`, binding the role back to
`lib/ontology/teaching_roles.py` and `lib/ontology/framework_blocks.py`. Entries may also declare
`profile_edges[]`. The `instructional` profile additionally declares a `grounds_in` list naming exactly
those two modules plus `schemas/ONTOLOGY.md`.

### L3 — the four block-ontology lexicons

Each is a flat marker table: `x-markers` is an array of `{marker, role, notes}`. Nothing else.

| File | Markers | Roles it triggers |
|---|---:|---|
| `openstax_lexicon.json` | 6 | `guided_practice`, `worked_example`, `exercise_item` |
| `generic_instructional_lexicon.json` | 10 | `guided_practice`, `exercise_item` |
| `federal_register_lexicon.json` | 4 | `agency_header`, `docket_line`, `enacting_clause` |
| `ansi_z535_lexicon.json` | 9 | `safety_notice`, `procedure_step`, `prerequisite`, `materials_list`, `troubleshooting_entry` |

The ANSI file is named for the Z535.6 severity words it opens with, but it is **not** a
`safety_notice`-only lexicon — it carries the `technical_manual` profile's procedural vocabulary too.

`exercise_apparatus_lexicon.json`, `objective_filler_lexicon.json`, and `semantik_lexicon.json` also
live under `schemas/taxonomies/` but belong to other subsystems, not to this ontology.

### Relations — `schemas/taxonomies/block_relations.json`

`x-relations` carries **14** entries across two families (`structural`, `profile`):

```
same_unit  continues  caption_of  adjacent  same_section  footnote_of  refers_to
same_story  solution_of  practice_of  answers  cites  defines  references
```

Relations are the escape valve for genuinely n-ary structure. A `worked_example` block does not
*contain* its solution as a second label — it is `solution_of`-linked to a separate `solution` block.
**Structure that is not a single block's identity is an EDGE, not a second label.** Two edges exist
specifically because no block identity could carry them: `same_story` (plus a `story_id` grouping) ties
the blocks of one content thread on a multi-story page, composing with `continues` for a story that
jumps pages; and `refers_to` binds a citing block to the figure/chart/table/heading/math_block target of
an in-text numbered cross-reference.

---

## 3. Policies the data encodes

### Multi-label policy: one kind, at most one role

- **One kind** — the coverage/conservation invariant (every unit typed exactly once) is what lets a
  converter assert no content was lost. A multi-kind block breaks it.
- **At most one role** — roles are mutually exclusive *within a resolved profile*: a block is a
  `worked_example` **or** a `solution`, not both. If two markers match, the more specific / earliest-
  anchored wins (source-specific lexicon before generic; longest match before shortest). Cross-profile
  ambiguity does not arise because a document resolves one profile.

### Graceful degradation: no profile → Layer-1 only

A document with no matching genre profile still gets a **complete labeling** — every block resolves an
L1 kind, roles are simply absent. Degradation is also **per-block**: inside an instructional textbook, a
plain body paragraph that matches no role stays L1-`paragraph` while the worked example beside it
carries a role. There is no "unknown role" bucket — **absence of a role is the honest signal**. A
brand-new genre ships useful L1 output on day one and gets richer when someone writes its profile.

### Authority ladder: labels earn trust

A label produced by a *learned* head (rather than deterministic markup) starts as a **hint** (a `data-*`
breadcrumb), graduates to a **validator** (proposes; deterministic code disposes) once calibrated across
≥2 corpora, and only then joins the bulk path — always under fail-closed conservation invariants. The
ontology defines the label space; it does not grant a model authority to apply it.

---

## 4. Governance

| Change | Layer | Gate | Blast radius |
|---|---|---|---|
| Add / remove / rename a **kind** | L1 | schema-change sign-off (maintainer-gated, not a data PR) | new classifier head class, new DocLayNet gap, re-train, conservation-contract review |
| Add a **role** to a profile | L2 | maintainer (data PR) — must declare `attaches_to` ≥1 kind | additive; no head change |
| Add a **genre profile** | L2 | maintainer (data PR) | additive |
| Add a **lexicon entry / file** | L3 | maintainer (data PR) — pure vocabulary | additive; zero code |
| Add / rename a **relation** | relations | schema-change sign-off if it becomes a trained head class; else maintainer | may add a head class |

### Mechanically enforced invariants

`schemas/tests/test_block_ontology.py` contains seven test functions. What each actually guards:

| Test | Guards |
|---|---|
| `test_ontology_file_loads` | every ontology file loads through `load_taxonomy` |
| `test_block_kinds_declare_doclaynet_key` | every L1 kind declares a `doclaynet` key (explicit `null` permitted, with a gap note) |
| `test_l1_kind_enum_matches_expected_snapshot` | the L1 enum matches the in-test `EXPECTED_L1_KINDS` snapshot |
| `test_roles_attach_to_valid_l1_kinds` | every L2 role's `attaches_to` names only real L1 kinds |
| `test_lexicon_is_marker_to_role_only` | no lexicon carries anything but marker→role rows — i.e. no publisher marker escapes L3 |
| `test_profile_relations_name_real_roles` | every profile relation names ≥1 real L2 role |
| `test_relation_enums_match_entries_by_family` | the relation enums agree with `x-relations` entries, per family |

The snapshot test is what makes "closed L1" a mechanical fact rather than a convention: adding a kind
fails CI until the snapshot is deliberately updated.

**One invariant is review-only, with no test.** The intended rule "every current arranger `TYPE_ENUM`
value and every onboarding-aligner `SOURCE_TYPE_MAPS` output has a `(kind[, role])` home" cannot be
tested here, because `SOURCE_TYPE_MAPS` lives in an out-of-tree workspace that CI in this repository
cannot see. Do not assume CI is watching that door.

---

## 5. Adoption status

This is the section to read before citing anything above as shipping behavior.

### What is consumed today

**The L3 lexicon layer has one production consumer.**
`SemantiK/semantik_structure/glmocr/region_map.py` — part of the GLM-OCR extraction lane — reads
`schemas/taxonomies/openstax_lexicon.json`, walking its `x-markers` rows to build the
`(marker, pedagogy_css_class)` table that drives apparatus classification. It maps L2 role names
(`guided_practice`, `worked_example`, `exercise_item`) onto pedagogy CSS classes via an in-module
`_ROLE_TO_CSS` dict, unions the schema markers with a frozen in-module fallback (so a bare SemantiK
checkout still works), and sorts longest-first so a specific marker beats a generic prefix.

Two properties of that reader matter for anyone changing the lexicons:

- It does **not** go through `load_taxonomy`. It locates `schemas/taxonomies` by walking up parent
  directories from its own file, and returns `None` when the tree is unreachable. This is deliberate:
  the SemantiK runtime may execute in a separate venv without the Ed4All schema tree on the path.
- Failures are silent by design (`except (OSError, ValueError, TypeError): pass` → frozen fallback).
  A malformed lexicon degrades to the fallback rather than failing the conversion, so a bad data PR will
  not announce itself loudly. Validate lexicon edits with the schema test suite.

The same module separately reads `semantik_lexicon.json`'s per-profile `apparatus_sections` display
lists for end-matter section headings — deliberately *not* the broader `apparatus_whitelist`, because
promoting bare worked-example body labels or callout labels to section headings is exactly the
over-segmentation the furniture constraint fights.

### What is not consumed

**No production code path reads `block_kinds.json`, `block_relations.json`, or any
`genre_profile_*.json`.** Grepping the tree for those filenames finds only
`schemas/tests/test_block_ontology.py`. The L1 and L2 layers are a ratified, test-enforced vocabulary
with no consumer.

**Three of the four L3 lexicons have no consumer either.** `generic_instructional_lexicon.json`,
`federal_register_lexicon.json`, and `ansi_z535_lexicon.json` are read by nothing outside the schema
tests. Only `openstax_lexicon.json` is live.

### The two concrete gaps

1. **The page-arranger contract still uses its own flat enum.**
   `SemantiK/semantik_structure/page_arranger_contract.py` carries `CONTRACT_VERSION = 2` and a
   `TYPE_ENUM` frozenset of 9 values — `heading`, `paragraph`, `table`, `figure_caption`, `example`,
   `solution`, `exercise_list`, `definition_box`, `furniture` — plus a `TYPE_ALIASES` read-compat map.
   It conflates structural kind with pedagogical role. The § 6 migration to `(kind, role)` tuples has
   not happened.

2. **The extraction lane carries a parallel vocabulary.** `region_map.py` maps the layout model's
   25-class `native_label` (the PP-DocLayoutV3 taxonomy — the *input*) onto its own `region_kind` set:
   `heading`, `paragraph`, `figure`, `table`, `math`, `caption`, `footnote`, `aside`, `metadata_drop`
   (plus `list` per its docstring). That set is neither L1 nor the arranger enum. Reconciling it with
   L1 is unscheduled and is the single largest gap between this document and the shipping converter.

**A caveat on the source mappings below.** `SOURCE_TYPE_MAPS` and `PRACTICE_MARKER_LEXICON`, referenced
throughout § 6 as "the current map", exist only in an out-of-tree workspace. They are **not** in this
repository, and statements about what they "currently" do are unverifiable from this tree.

### Sequenced follow-up

1. Add typed loaders under `lib/ontology/` (mirroring `teaching_roles.py`) so consumers get a checked
   surface instead of raw dicts.
2. Reconcile the lane's `region_kind` set against L1.
3. Only then wire the arranger contract to read these files instead of its in-code enum.

---

## 6. Planned mapping (not yet implemented)

Everything in this section describes intended future behavior. None of it runs.

### 6.1 Arranger `TYPE_ENUM` → `(L1 kind, L2 role)`

The projection is total (every enum value has a home) and is a pure relabeling — byte-compatible on
read.

| arranger `TYPE_ENUM` | L1 kind | L2 role (profile) | Note |
|---|---|---|---|
| `heading` | `heading` | — | `level` attr preserved; document title → `title_block` |
| `paragraph` | `paragraph` | — | the fallback kind |
| `table` | `table` | — | 1:1 |
| `figure_caption` | `caption` | — | splits: a real image becomes `figure` + `caption` |
| `example` | `paragraph` | `worked_example` (instructional) | kind is prose; role names the pedagogy |
| `solution` | `paragraph` | `solution` (instructional) | " |
| `exercise_list` | `list_item` | `exercise_item` (instructional) | list granularity is per-item in L1 |
| `definition_box` | `paragraph` | `definition_box` (instructional) | also valid on `blockquote` |
| `furniture` | `furniture` | — | `subkind` attr (running_header / page_number / …) |

Four values are pure kinds and map losslessly; four pedagogical values *gain* a role while their kind
stays unambiguous; `figure_caption` is the one rename, and it comes with a split.

### 6.2 What a v3 arrangement block would carry

```json
{"ids": ["p3_u12", "p3_u13"],
 "kind": "paragraph",
 "role": "worked_example",
 "profile": "instructional",
 "level": null,
 "subkind": null,
 "continues_prev_page": false}
```

Changes from `CONTRACT_VERSION = 2`:

1. **`type` → `kind` + `role` + `profile`.** `kind` is a closed L1 value; `role`/`profile` are optional
   (absent ⇒ Layer-1-only, the graceful-degrade path). A v2 `type` value maps in via § 6.1 on read.
2. **Furniture is LABELED, never dropped.** v2 lets the teacher omit furniture units; v3 would require
   every furniture unit in a `furniture` block with a `subkind`. The coverage invariant already forbids
   dropped ids; this makes furniture a first-class labeled kind.
3. **`figure_caption` splits** into `figure` + `caption`, bound by `caption_of`.
4. **Nine kinds become expressible** that v2's 9 values could not name: `math_block`, `code_block`,
   `blockquote`, `footnote`, `form_field`, `title_block`, `separator`, `aside`, `chart` — all of which
   currently fall to `paragraph` or `figure`.
5. **`CONTRACT_VERSION` 2 → 3**, with the existing `TYPE_ALIASES` table preserved as the read-compat
   shim.

### 6.3 Source-genre mappings

Which L2 profile applies to each inventoried gold source, and what its native markup gives L1. The
"current map" column describes the out-of-tree aligner and is unverifiable from this tree.

| Source | L2 profile | Native structure signal | Current map state |
|---|---|---|---|
| arxiv | `scholarly` | LaTeXML `ltx_*` classes | has class map |
| wikipedia | `encyclopedic` | Parsoid tag + class | has class map |
| openstax | `instructional` | `data-type` + `os-*` | class + datatype + fine map; **L3 lexicon live** |
| pmc | `scholarly` | JATS-derived HTML tags | generic-tag fallback only |
| cfr | `legal_regulatory` | eCFR HTML tags | generic-tag fallback only |
| federal_register | `legal_regulatory` | field-label lines | generic + L3 lexicon |
| courtlistener | `legal_regulatory` | opinion HTML tags | generic-tag fallback only |
| nces_digest | `statistical` | statistical tables | generic-tag fallback only |
| gutenberg | `literary` | literary prose | generic-tag fallback only |
| forms | `forms` | field / label layout | generic-tag fallback only |
| mkdocs-site | `instructional` | clean mkdocs HTML | via `ed4all import-docs` |
| manuals | `technical_manual` | ANSI Z535 signal words + step numbering | profile + L3 lexicon; no gold source inventoried |
| infographics | *(none → L1 only)* | figure-dominant | figure path |

Recurring gaps across sources, in rough priority order:

- **`equation` → `math_block`.** Both the arxiv (`ltx_equation`, `ltx_equationgroup`) and openstax
  (`equation`) maps currently coerce display math to `paragraph`. This is the single most-repeated
  mapping defect and defeats the point of having a `math_block` kind at all.
- **`ltx_theorem` → `(paragraph, theorem[scholarly])`**, not the instructional `definition_box` it
  currently coerces to. Same shape, different genre.
- **No class map at all** for pmc, cfr, courtlistener, nces_digest, gutenberg, and forms — all six fall
  through to the generic-tag fallback, so their declared L2 roles resolve only when a heading or marker
  heuristic happens to name them. For pmc the signal exists and is unused (JATS `sec-type` /
  `article-*` attributes); for courtlistener and gutenberg there is genuinely no markup signal, and the
  roles need positional heuristics or an L3 lexicon.
- **`form_field` is never minted.** Form detection needs a geometry/layout signal (label-value
  adjacency, checkbox glyphs) that an HTML-truth aligner cannot see. It is also the one L1 kind with no
  DocLayNet parent, so weak supervision cannot bootstrap it either. This is a vision/geometry
  onboarding step.
- **`PRACTICE_MARKER_LEXICON` must move out of code** into `openstax_lexicon.json` +
  `generic_instructional_lexicon.json`. The aligner's practice-marker fine-typer should read the lexicon
  files rather than a hardcoded dict — the wide-net rule applied to the aligner. The class maps
  (`ltx_*` → kind) are structural, not publisher vocabulary, so those may remain source-adapter code.

The generic-tag fallback itself needs no change: it already emits L1 kinds (`hN`→heading,
`table`→table, `figcaption`→caption, `li`→list_item, `dt`/`dd`→`(paragraph, definition_box)`).

For `infographics`, note that the semantic content lives *inside* the image, so VLM alt-text and
extended description are load-bearing. The block ontology labels the figure envelope, not the
infographic's internal structure.
