# Learner Answer UI — Manual Accessibility Pass

Operator procedure for the **per-cycle** manual screen-reader + keyboard
accessibility verification of the learner answer surface (`/learn/`). The
automated WCAG 2.2 AA gate (`gui/tests/test_learner_a11y_gate.py`) runs every CI
build over the server-rendered HTML; this manual pass covers what a static
validator cannot: live screen-reader announcements, real keyboard focus order,
zoom reflow, High-Contrast rendering, and assistive-technology (AT) behavior on a
real browser.

Run this pass **once per release cycle** and any time the learner page shell,
the answer-rendering fragment, the source-viewer wrapper, or the focus/live-region
JavaScript changes. File a defect per failure, fix, and **re-run the affected
section** before sign-off.

The conformance bar is **WCAG 2.2 Level AA**: any Level A or AA failure blocks
sign-off. Level AAA observations are recorded as notes, not blockers.

---

## 0. Setup

1. Launch the learner surface on a trusted local machine:

   ```bash
   ed4all gui --learner          # learner-only mount; / is the learner page
   # or, for the full app:  ed4all gui   →  open /learn/
   ```

   Confirm the default loopback bind (`127.0.0.1:8077`). Never run this pass
   against a routable bind of the **full** app.

2. Seed at least one answerable course so the ask flow returns real data,
   including at least one course whose citations resolve to **both** a converted
   HTML source page (the SemantiK `semantik:` provenance source) and an IMSCC member
   page (so the source-viewer route is exercised on both resolution paths).

3. Have ready a set of queries that deterministically drive each answer status
   (see § 1.2). If a status cannot be driven from real data this cycle, note it
   and verify that status from the automated gate's rendered fixture instead.

4. AT + browser matrix for this cycle (record exact versions in the sign-off
   table):

   | AT | Browser | OS |
   |----|---------|----|
   | NVDA | Firefox | Windows |
   | VoiceOver | Safari | macOS |
   | (keyboard only — no AT) | any | any |

5. Install the **axe DevTools** browser extension (no axe/playwright dependency
   is added to the repo — the sweep is manual, per cycle).

---

## 1. Rendering contract under test

The page must behave identically for each answer outcome. The fragment skeleton
is the same bones for every status; only the heading copy and body change:

```html
<section class="answer" data-status="{status}" aria-labelledby="answer-h">
  <h2 id="answer-h" tabindex="-1">{heading}</h2>
  {body}
</section>
```

### 1.1 Status → expected heading / body

| Status / error key | Heading | Body the AT must convey |
|--------------------|---------|-------------------------|
| `answered` | "Answer" | Answer paragraphs, then an `<h3>Sources</h3>` ordered list of `Source: {page_label}` links. |
| `answered_with_warnings` | "Answer" | Same as `answered`, preceded by an advisory note ("Parts of this answer may not be fully supported…"). |
| `refused_low_confidence` | "No answer found" | "We couldn't find course material that answers this question confidently." + rephrase guidance. **No** answer text, **no** sources. |
| `refused_not_in_course` | "Not covered in this course" | "This doesn't appear to be covered in this course's materials." + guidance. **No** answer text, **no** sources. |
| `blocked_invalid_citation` / `blocked_citation_gate` | "Answer withheld" | "We found a possible answer but couldn't verify it… so it wasn't shown." **No** answer text, **no** sources. |
| `error_backend_down` | "The answer engine isn't available" | "The local answer engine isn't running. Please tell the session facilitator." |
| `error_misconfigured` / `error_index` / `error_compose` / `error_generic` | "Something went wrong" | "The course assistant hit a problem and couldn't answer. Please tell the session facilitator." |

Hard rules to verify by ear/eye on every status:

- Refusal/blocked/error states show **no** citation list and **no** answer text.
- No raw internals (confidence, latency, model id, groundedness, exception text)
  appear in the rendered HTML — those live in the JSON only.
- Every `Source:` citation is a real focusable link with full text
  ("Source: {page_label}"), never an image and never a bare URL.

### 1.2 Driving each status

| Status | How to drive it |
|--------|-----------------|
| `answered` | A question well-covered by the seeded course. |
| `answered_with_warnings` | A question that retrieves partially-supported passages (low groundedness path), if reproducible this cycle. |
| `refused_low_confidence` | A vague/ambiguous in-domain question. |
| `refused_not_in_course` | A clearly off-topic question. |
| `blocked_*` | A question that trips the citation gate (rare; use the automated fixture if not reproducible). |
| `error_backend_down` | Stop the local answer engine (Ollama) and ask. |
| other errors | If not reproducible live, verify the rendered copy from the automated gate fixture. |

---

## 2. Keyboard-only walkthrough (no AT)

Use **Tab** / **Shift+Tab** / **Enter** / **Space** only. A **visible focus
indicator must be present at every stop** (a real outline, not a removed one).
No focus traps; focus order must be logical (DOM order).

| # | Step | Expected |
|---|------|----------|
| K1 | Load `/learn/`, press Tab once | Focus lands on the **"Skip to main content"** skip link; it is visible when focused. |
| K2 | Enter on the skip link | Focus jumps into `<main>` (the question heading / first control), bypassing the header. |
| K3 | Tab through the form | Order: **Course** select → **Your question** textarea → **Ask** button. Each control is reachable and has a visible focus ring. |
| K4 | Operate the **Course** select by keyboard | Options selectable with arrows/Enter; no mouse needed. |
| K5 | Type a question, Tab to **Ask**, press Enter (or Space) | Form submits; the form disables during the request; a polite busy state is set. |
| K6 | After the answer arrives | Focus has **moved to the answer heading** (`#answer-h`, `tabindex="-1"`). |
| K7 | Tab from the answer heading (for `answered`) | Reach each **"Source: …"** link in order; each has a visible focus ring. |
| K8 | Activate a `heading`-fragment source link | The source viewer opens and lands on the **cited heading** (`#slug` fragment); for `xpath`/no-fragment links it lands at top-of-page with the banner note. |
| K9 | In the viewer, Tab | Reach the **"Back to your question"** link in the banner nav; activating it returns to the question. |
| K10 | Throughout | **No keyboard trap** anywhere; Shift+Tab reverses cleanly; Escape never strands focus. |
| K11 | Target size | All interactive controls are ≥ 24×24 px (spot-check; WCAG 2.5.8). |

**Pass = every row behaves as expected with a visible focus indicator and no
trap.**

---

## 3. Screen-reader scripts

Run the full script under **each** AT/browser pair in the § 0 matrix. The
single live region is `<p id="status" role="status" aria-live="polite">`; the
visual-only elapsed counter is `aria-hidden` and must **never** be announced.

### 3.1 NVDA + Firefox (Windows) / VoiceOver + Safari (macOS)

| # | Action | Expected announcement / behavior |
|---|--------|----------------------------------|
| S1 | Navigate to the page | Page title "Ask the Course"; document `lang` is English. |
| S2 | Move to `<main>` | The `<h1>` "Ask the course a question" is reachable as the page's single top-level heading. |
| S3 | Tab to the **Course** control | Announced as a combobox/select **with its "Course" label**. |
| S4 | Tab to the **question** field | Announced as a multi-line edit **with its "Your question" label**. |
| S5 | Submit | The busy message ("Searching the course materials. This can take up to a minute.") is announced **exactly once**. The per-second elapsed counter is **not** announced. |
| S6 | On arrival | The arrival message is announced **once** ("Answer ready." / "No answer found." / the error copy). Focus moves to the answer heading and the heading text is announced. |
| S7 | Browse mode read-through | The answer body is fully readable in browse/read mode; the `Sources` `<h3>` and the ordered list are reached as a list of links. |
| S8 | Each source link | Announced with the **full** "Source: {page label}" text (not "link" alone, not the URL). Module / "(approximate location)" annotations are announced as text. |
| S9 | Activate a source link → viewer | The viewer banner is announced as a navigation region labeled "Source context"; the **archived page's own `<h1>`** remains the page's single top-level heading (the banner adds no heading). |
| S10 | Fragment navigation | For a `heading` fragment, the cited section heading is where reading resumes / focus lands. |
| S11 | "Back to your question" | Announced as a link; activating it returns to the question page. |
| S12 | Refusal/blocked/error states | The arrival message is announced once; the heading + guidance copy read cleanly; **no** phantom "Sources" list is present. |

**Pass = all announcements occur as described, the busy/arrival messages each
fire exactly once, and the elapsed counter is silent.**

---

## 4. axe DevTools sweep

Run the axe DevTools extension on `/learn/` in **each** of these page states and
record the violation count (the bar is **zero** Level A/AA violations):

| State | How to reach it |
|-------|-----------------|
| Idle | Fresh page load, before asking. |
| Busy | Mid-request (the form is disabled, `aria-busy="true"`, status set). Pause the backend or use a slow course to capture this. |
| Answered | An `answered` (or `answered_with_warnings`) result rendered. |
| Refused | A `refused_*` result rendered. |
| Blocked | A `blocked_*` result rendered (or note as gate-only this cycle). |
| Error | An error result rendered (e.g. backend down). |
| Source viewer | A source page opened from a citation (both `semantik:`-provenance SemantiK HTML and IMSCC sources). |

---

## 5. Reflow / contrast / High-Contrast spot checks

| # | Check | Expected | WCAG |
|---|-------|----------|------|
| Z1 | Browser zoom to **200 %** (and 400 % at 1280px width) | No horizontal scroll for primary content; nothing clipped or overlapping; the form and answer remain usable. | 1.4.10 Reflow |
| Z2 | Text contrast (sample headings, body, links, status text) with a contrast checker | ≥ 4.5:1 for normal text, ≥ 3:1 for large text and UI component boundaries / focus indicators. | 1.4.3 / 1.4.11 |
| Z3 | **Windows High Contrast Mode** (Forced Colors) | Focus outline still visible; text and controls still legible; no `outline:none`-only controls; busy/error state still conveyed by text (not color alone). | 1.4.1 / 2.4.7 |
| Z4 | `prefers-reduced-motion: reduce` set | Any spinner/animation is suppressed or reduced. | 2.3.3 |
| Z5 | Target size spot-check at default zoom | Interactive targets ≥ 24×24 px. | 2.5.8 |

---

## 6. Defect loop

For every failure in §§ 2–5:

1. File a defect with: AT/browser/OS + version, page state, the WCAG criterion,
   the observed vs. expected behavior, and a repro.
2. Fix in the owning module (the learner page shell, the answer-render fragment,
   the source-viewer wrapper, or the focus/live-region JS).
3. **Re-run the affected section** of this pass.
4. Sign off only when every section passes with zero open Level A/AA defects.

---

## 7. Sign-off log

Record one row per AT/browser run per cycle. The cycle is signed off only when
every applicable row is **PASS** with no open Level A/AA defect.

| Date | Cycle / release | AT | AT version | Browser | Browser version | OS | Sections run | Result (PASS/FAIL) | Open defects | Notes / tester |
|------|-----------------|----|-----------|---------|-----------------|----|--------------|--------------------|--------------|----------------|
|      |                 | NVDA |         | Firefox |                 | Windows | §2–§5 |                    |              |                |
|      |                 | VoiceOver |    | Safari  |                 | macOS | §2–§5 |                    |              |                |
|      |                 | keyboard-only | — | —    | —               | —  | §2 |                    |              |                |
|      |                 | axe DevTools | — | —      |                 |    | §4 |                    |              |                |

> **Privacy reminder:** learner queries are logged locally to
> `runtime/training-captures/` (JSONL, via `DecisionCapture`) on the same device, with no
> telemetry path. Disclose this in the session consent language before any pilot
> run that records real participant queries.
