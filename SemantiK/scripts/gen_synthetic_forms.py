"""Generate (features → StructuredBlock[] JSON) training pairs from synthetic forms.

This is the first IR-based parser. It exercises the full stack end-to-end:
    random spec  ->  ir.Document(Form)  ->  emit_html(doc)  ->  axe-core gate
                                                                     |
                                                    drop on fail  <--+
                                                                     |
                                                                     v
                                      render HTML -> PDF -> pdf_to_ocr_text -> input_ocr
                                                                     |
                                         output_html kept as ground truth for classifier labels
                                                                     |
                                                            write pair JSON

Forms are chosen as the first generator because (a) WCAG criteria for forms
are concrete and enforceable, (b) the synthetic surface is small enough to
cover exhaustively, and (c) failing examples surface emitter bugs fast.

Usage:
    python scripts/gen_synthetic_forms.py --n 10 --out-dir data/synthetic/forms
"""
from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from dart_semantic import emit_html, ir  # legacy IR + emitter for ground-truth HTML
from dart_semantic.features import pdf_to_ocr_text
from dart_semantic.validate import HtmlValidator

# CSS ensures the rendered PDF actually meets WCAG 2.2 SC 2.5.8 (target size):
# every interactive control is >= 24 x 24 CSS px.
CSS = """
body { font-family: Arial, sans-serif; max-width: 7in; margin: 1in auto; color: #111; }
h1 { margin: 0 0 0.4em; }
form { margin-top: 1em; }
fieldset { border: 1px solid #888; padding: 0.8em 1em; margin: 0.8em 0; }
legend { font-weight: bold; padding: 0 0.4em; }
.field { margin: 0.6em 0; }
label { display: block; margin-bottom: 0.2em; font-weight: 600; }
input, select, textarea {
    min-width: 24px; min-height: 24px;
    padding: 4px 6px; font-size: 14px;
    border: 1px solid #666;
}
input[type=checkbox], input[type=radio] { min-width: 24px; min-height: 24px; margin-right: 0.4em; }
textarea { width: 100%; height: 4em; }
"""

FORM_TEMPLATES = [
    # ---- Education ----
    ("Student Registration", [
        ("personal", "Personal Information", [
            ("full_name",    "Full name",        "text",     True),
            ("date_of_birth","Date of birth",    "date",     True),
            ("email",        "Email address",    "email",    True),
            ("phone",        "Phone number",     "tel",      False),
        ]),
        ("academic", "Academic Information", [
            ("program",      "Program",          "select",   True,
                ["Undergraduate", "Graduate", "Professional", "Continuing education"]),
            ("start_term",   "Starting term",    "select",   True,
                ["Fall 2026", "Spring 2027", "Summer 2027"]),
            ("full_time",    "Enrollment status","radio",    True,
                ["Full-time", "Part-time"]),
        ]),
        ("accommodations", "Accommodation Requests", [
            ("accommodations", "Do you require any of the following?", "checkbox", False,
                ["Large print materials", "Sign language interpreter",
                 "Extended testing time", "Accessible seating"]),
            ("notes",       "Additional notes", "textarea", False),
        ]),
    ]),

    # ---- Higher ed accessibility workflows ----
    ("ADA Accommodation Request", [
        ("requester", "Requester Information", [
            ("name",        "Name",             "text",     True),
            ("email",       "Contact email",    "email",    True),
            ("role",        "Role",             "select",   True,
                ["Student", "Faculty", "Staff", "Visitor"]),
            ("id_number",   "University ID",    "text",     False),
        ]),
        ("request", "Request Details", [
            ("needs",       "Type of accommodation needed", "checkbox", True,
                ["Physical access", "Visual aids", "Auditory aids",
                 "Cognitive support", "Service animal", "Assistive technology"]),
            ("details",     "Describe your request",        "textarea", True),
            ("needed_by",   "Date needed by",               "date",     False),
        ]),
    ]),

    # ---- K-12: IEP intake ----
    ("IEP Intake Form", [
        ("student", "Student Information", [
            ("student_name","Student name",     "text",     True),
            ("dob",         "Date of birth",    "date",     True),
            ("grade",       "Current grade",    "select",   True,
                ["K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]),
            ("school",      "School",           "text",     True),
        ]),
        ("guardian", "Parent / Guardian", [
            ("guardian_name","Parent or guardian name","text", True),
            ("relationship", "Relationship to student", "select", True,
                ["Parent", "Legal guardian", "Foster parent", "Grandparent", "Other"]),
            ("email",       "Contact email",    "email",    True),
            ("phone",       "Contact phone",    "tel",      True),
        ]),
        ("disability", "Eligibility Categories", [
            ("categories",  "Which categories apply?", "checkbox", False,
                ["Specific learning disability", "Speech or language impairment",
                 "Intellectual disability", "Emotional disturbance",
                 "Autism spectrum disorder", "Hearing impairment",
                 "Visual impairment", "Orthopedic impairment",
                 "Other health impairment", "Multiple disabilities",
                 "Deaf-blindness", "Traumatic brain injury",
                 "Developmental delay"]),
            ("concerns",    "Parent/guardian concerns", "textarea", True),
        ]),
    ]),

    # ---- K-12: 504 plan ----
    ("Section 504 Plan Request", [
        ("student", "Student", [
            ("student_name","Student name",     "text",     True),
            ("dob",         "Date of birth",    "date",     True),
            ("school",      "School",           "text",     True),
        ]),
        ("condition", "Disability or Condition", [
            ("diagnosis",   "Documented diagnosis or condition", "textarea", True),
            ("impact",      "Major life activity impacted", "checkbox", True,
                ["Learning", "Walking", "Seeing", "Hearing", "Speaking",
                 "Breathing", "Eating", "Sleeping", "Concentrating",
                 "Reading", "Thinking", "Communicating"]),
        ]),
        ("accommodations", "Requested Accommodations", [
            ("in_classroom","Classroom accommodations", "textarea", False),
            ("testing",     "Testing accommodations",   "textarea", False),
            ("physical",    "Physical environment accommodations", "textarea", False),
        ]),
    ]),

    # ---- Employment: reasonable accommodation request ----
    ("Reasonable Accommodation Request (Employment)", [
        ("employee", "Employee", [
            ("name",        "Employee name",    "text",     True),
            ("employee_id", "Employee ID",      "text",     True),
            ("position",    "Position title",   "text",     True),
            ("department",  "Department",       "text",     True),
            ("supervisor",  "Supervisor name",  "text",     True),
        ]),
        ("request", "Accommodation Request", [
            ("limitations", "Functional limitation(s)", "textarea", True),
            ("requested",   "Accommodation(s) requested", "textarea", True),
            ("duration",    "Expected duration", "select", False,
                ["Permanent", "Temporary — less than 3 months",
                 "Temporary — 3 to 12 months", "Intermittent"]),
        ]),
    ]),

    # ---- Feedback / survey (DART product feedback etc.) ----
    ("Feedback Survey", [
        ("rating", "Service Rating", [
            ("satisfaction","Overall satisfaction", "radio", True,
                ["Very satisfied", "Satisfied", "Neutral",
                 "Dissatisfied", "Very dissatisfied"]),
            ("likelihood", "Likelihood to recommend", "select", False,
                ["Very likely", "Likely", "Neutral", "Unlikely", "Very unlikely"]),
        ]),
        ("comments", "Your Comments", [
            ("helped",      "What worked well?", "textarea", False),
            ("improve",     "What could we do better?", "textarea", False),
        ]),
    ]),

    # ---- Event / training registration ----
    ("Event Registration", [
        ("attendee", "Attendee", [
            ("name",        "Full name",        "text",     True),
            ("email",       "Email",            "email",    True),
            ("company",     "Organization",     "text",     False),
            ("title",       "Job title",        "text",     False),
        ]),
        ("session", "Session Preferences", [
            ("track",       "Session track",    "select",   True,
                ["Technical", "Business", "Policy", "Research"]),
            ("dietary",     "Dietary needs",    "checkbox", False,
                ["Vegetarian", "Vegan", "Gluten-free", "Kosher", "Halal",
                 "Dairy-free", "Nut-free", "None"]),
            ("comments",    "Special requests", "textarea", False),
        ]),
    ]),

    # ---- Grievance / complaint form ----
    ("ADA Grievance Form", [
        ("complainant", "Complainant Information", [
            ("name",        "Your name",        "text",     True),
            ("email",       "Email address",    "email",    True),
            ("phone",       "Phone number",     "tel",      False),
            ("anonymous",   "Submit anonymously?", "radio", True, ["Yes", "No"]),
        ]),
        ("incident", "Incident Details", [
            ("date",        "Date of incident", "date",     True),
            ("location",    "Location",         "text",     True),
            ("description", "Describe the incident", "textarea", True),
            ("witnesses",   "Witnesses (if any)", "textarea", False),
        ]),
        ("resolution", "Requested Resolution", [
            ("resolution",  "What outcome are you seeking?", "textarea", True),
        ]),
    ]),
]


def _runs(text: str) -> list[ir.Run]:
    return [ir.Run(text)]


def _pick_fields(rng: random.Random, template_fields: list) -> list[ir.FormField]:
    """Drop a random subset of optional fields to introduce variation.
    Required fields are always kept; optional fields kept with p=0.75."""
    out: list[ir.FormField] = []
    for entry in template_fields:
        name, label, kind, required, *rest = entry
        if not required and rng.random() < 0.25:
            continue  # drop this optional field
        options = rest[0] if rest else []
        out.append(ir.FormField(
            kind=kind,
            name=name,
            label=_runs(label),
            required=required,
            options=list(options),
        ))
    return out


def gen_form_doc(rng: random.Random, variant_id: int) -> ir.Document:
    title, fieldsets_spec = rng.choice(FORM_TEMPLATES)
    # Vary the title slightly across variants so the model doesn't memorize it.
    title_suffix = rng.choice(["", " — 2026", " (Spring)", " (Fall)", " Form"])
    doc_title = f"{title}{title_suffix}"

    fieldsets = []
    for (fs_name, legend, fields_spec) in fieldsets_spec:
        fields = _pick_fields(rng, fields_spec)
        if not fields:
            continue
        fieldsets.append(ir.Fieldset(legend=_runs(legend), fields=fields))

    return ir.Document(
        title=doc_title,
        language="en",
        source="synthetic_form",
        source_id=f"form_{variant_id:05d}",
        blocks=[
            ir.Heading(level=1, runs=_runs(doc_title)),
            ir.Paragraph(runs=_runs(f"Please complete this {title.lower()} form.")),
            ir.Form(
                action="/submit",
                method="post",
                fieldsets=fieldsets,
            ),
        ],
    )


def wrap_html_document(body_html: str, title: str) -> str:
    """The emitter gives us the document; we only need to inject the CSS
    for the target-size styles to apply at render time."""
    # Inject our stylesheet after <head>
    return body_html.replace(
        "<head>",
        f"<head><style>{CSS}</style>",
        1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="How many forms to ATTEMPT (fewer will pass the gate).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/synthetic/forms"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    attempted = 0
    emitter_drops = 0
    axe_drops = 0
    ok = 0

    with HtmlValidator() as validator:
        for variant_id in range(args.n):
            attempted += 1
            doc = gen_form_doc(rng, variant_id)

            # 1. IR → HTML (may fail if our IR violates emitter invariants).
            try:
                raw_html = emit_html.emit(doc)
            except emit_html.EmitterError as e:
                emitter_drops += 1
                print(f"[{variant_id:04d}] emitter-drop: {e}")
                continue
            html_doc = wrap_html_document(raw_html, doc.title)

            # 2. Gate: axe-core with wcag22aa ruleset.
            result = validator.check(html_doc)
            if not result.ok:
                axe_drops += 1
                reasons = "; ".join(result.reasons[:3])
                print(f"[{variant_id:04d}] axe-drop: {reasons}")
                continue

            # 3. Render to PDF → layout-annotated OCR → input features.
            # Reuse the validator's open Chromium instance for rendering —
            # Python's sync Playwright can't be nested.
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / "form.pdf"
                validator.render_pdf(html_doc, pdf_path)
                input_ocr = pdf_to_ocr_text(pdf_path)

            # 4. Save pair. build_classifier_data reads output_html for labels.
            pair = {
                "source": "synthetic_form",
                "variant_id": doc.source_id,
                "title": doc.title,
                "input_ocr": input_ocr,
                "output_html": html_doc,
            }
            out_path = args.out_dir / f"{doc.source_id}.json"
            out_path.write_text(json.dumps(pair, ensure_ascii=False))
            ok += 1
            print(f"[{variant_id:04d}] ok — {doc.title} "
                  f"({len(input_ocr):>5} chars ocr / "
                  f"{len(html_doc):>5} chars html)")

    print(f"\n[summary] attempted={attempted} ok={ok} "
          f"emitter_drops={emitter_drops} axe_drops={axe_drops} "
          f"pass_rate={ok/attempted*100:.1f}%")


if __name__ == "__main__":
    main()
