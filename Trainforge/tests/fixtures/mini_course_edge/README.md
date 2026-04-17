# mini_course_edge — edge cases

Targets orthogonal to the seven main defects:

| File / key | Exercises |
|---|---|
| `course_objectives.json` | dual-IDs present; one orphan `w05-co-99` with **no** parent entry |
| `source_html/week_04_empty_terms.html` | JSON-LD `sections` present but `keyTerms` is `[]` (§4.4a H2) |
| `source_html/week_05_orphan_ref.html` | Chunk references `w05-co-99` → orphan pedagogical_scope_ref with `parent_id: null`, `status: "orphan"` |
| `source_html/week_05_tag_drift.html` | Concept tags carry both `contrast-minimum` and `contrast-minimum-level-aa`; the graph must collapse them to a single node |

Used by: `test_orphan_week_scoped_id_preserved_with_null_parent`, `test_contrast_minimum_concept_tags_collapse_to_single_node`.
