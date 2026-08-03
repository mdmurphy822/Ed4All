# Learner UI manual accessibility review

Use this guide to review the learner-facing question, answer, citation, and
source-viewer experience against WCAG 2.2 Level AA. Automated tests verify the
rendered contracts; this pass covers behavior that requires a browser,
keyboard, screen reader, zoom, or forced-colors environment.

Run the review before a release and after changes to the learner page shell,
answer fragments, source viewer, focus management, or live-region behavior.
Level A or AA failures block sign-off. Record review evidence in a private
issue or release system, not in the repository.

## Before you begin

1. Install the GUI dependencies using the
   [installation guide](installation.md).
2. Run the automated accessibility contracts:

   ```bash
   pytest -q gui/tests/test_learner_a11y_gate.py gui/tests/test_source_page.py
   ```

3. Start the learner-only surface on a trusted development system:

   ```bash
   ed4all gui --learner
   ```

   Open the learner URL reported by the command. For the full application,
   start `ed4all gui` and open its `/learn/` route. Do not expose a development
   instance to an untrusted network.
4. Use an operator-private test library with synthetic or appropriately
   licensed material. Do not place course identifiers, learner questions,
   screenshots, source pages, or answer captures in tracked files.
5. Prepare representative checks using:

   - keyboard-only navigation;
   - a desktop screen reader and a browser it supports;
   - browser zoom and narrow-viewport reflow;
   - forced-colors or high-contrast mode;
   - reduced-motion preferences; and
   - a browser accessibility analyzer for static findings.

Record the browser, assistive technology, operating system, and versions in
the private review record so failures can be reproduced.

## Outcomes to cover

Exercise every outcome that can be produced safely from the private test
library. Use the automated fixtures for outcomes that cannot be reproduced
without changing service state.

| Outcome | Required presentation |
|---|---|
| `answered` | An “Answer” heading, readable answer content, a “Sources” heading, and an ordered list of descriptive source links |
| `answered_with_warnings` | The answered presentation plus a plain-language advisory that is not communicated by color alone |
| `refused_low_confidence` | A “No answer found” heading and useful rephrasing guidance; no answer text or source list |
| `refused_not_in_course` | A “Not covered in this course” heading and useful guidance; no answer text or source list |
| `blocked_invalid_citation` or `blocked_citation_gate` | An “Answer withheld” heading and verification guidance; no candidate answer or source list |
| `error_backend_down` | A plain-language unavailable-service message with no implementation detail |
| Other typed or generic errors | A plain-language failure message with no exception, model, endpoint, confidence, or latency detail |

For all outcomes, confirm that the answer region has one programmatically
associated heading and that arrival focus moves to it. Source links must expose
meaningful link text rather than a bare URL.

## Keyboard review

Complete the following using only Tab, Shift+Tab, Enter, Space, arrow keys,
and Escape where applicable:

1. Load the learner page. The first Tab reveals the “Skip to main content”
   link, and activating it bypasses repeated navigation.
2. Move through the form. The course selector, question field, and Ask button
   follow a logical order, expose visible focus, and remain operable without a
   pointer.
3. Submit a representative question. The Ask control communicates its busy
   state and cannot be activated repeatedly while the request is active.
4. When the result arrives, focus moves to the answer heading without adding
   the heading to the normal Tab order.
5. For an answered result, move through every source link in document order.
   Each link has a visible focus indicator and a descriptive accessible name.
6. Activate a source link. The source viewer opens at the cited section when a
   valid fragment exists, or presents the documented approximate-location
   notice when it does not.
7. Activate “Back to your question.” The learner returns to the question
   surface without being trapped or stranded.
8. Reverse through the complete route with Shift+Tab. Confirm there is no
   keyboard trap and no hidden interactive control receives focus.
9. Spot-check that interactive targets meet the WCAG 2.5.8 minimum target-size
   requirement.

## Screen-reader review

Run the question-to-source flow with the selected screen reader. Test an
answered result and at least one refusal, blocked, or error result.

Verify that:

- the document language and page title are announced correctly;
- the page has one clear top-level heading and recognizable landmarks;
- the course selector and question field are announced with their visible
  labels, roles, and states;
- submitting produces one polite busy announcement;
- visual progress decoration and elapsed-time updates are not repeatedly
  announced;
- completion produces one arrival announcement and moves focus to the answer
  heading;
- warning, refusal, blocked, and error meaning is available in text rather
  than color or iconography alone;
- answered content, the Sources heading, list structure, and full source-link
  names are available in reading order;
- refusal, blocked, and error outcomes expose no phantom answer or Sources
  region;
- the source viewer announces its “Source context” navigation region while the
  archived page retains the primary page heading; and
- “Back to your question” is announced as a link and returns to the learner
  surface.

## Reflow, contrast, and motion

Review the idle, busy, answered, refused or blocked, error, and source-viewer
states.

- At 200% zoom, and at the WCAG reflow test condition of 400% at a 1280 CSS
  pixel viewport, primary content remains readable and operable without
  two-dimensional scrolling.
- Normal text, large text, controls, and focus indicators meet the applicable
  WCAG contrast requirements.
- In forced-colors or high-contrast mode, focus remains visible and state is
  never communicated by color alone.
- With reduced motion requested, progress and transition effects are removed
  or reduced without hiding status information.
- At the narrowest supported viewport, form controls, answer content, source
  links, and viewer navigation do not clip, overlap, or obscure one another.

Run the browser accessibility analyzer in each state. The release criterion is
zero unresolved Level A or AA violations. Treat automated output as evidence,
not a substitute for keyboard and screen-reader review.

## Source-viewer safety and accessibility

Open citations that represent each source form available in the private test
library. Confirm that:

- the viewer shows only the requested archived source content;
- navigation and cited-location behavior work with keyboard and screen reader;
- heading order remains understandable;
- images retain appropriate alternative text;
- tables remain associated with their headers;
- mathematical content has an accessible representation; and
- no local path, course identifier beyond the selected learner context,
  exception text, model detail, or service credential appears in the page.

The canonical learner routes and source-viewer security contract are described
in the [GUI learner-surface reference](../../gui/README.md#learner-surface-learn).

## Defects and sign-off

For each failure, record privately:

- the browser, assistive technology, operating system, and versions;
- the page state and action that produced the failure;
- the relevant WCAG criterion;
- expected and observed behavior; and
- a content-sanitized reproduction procedure.

Fix the owning learner-shell, answer-rendering, source-viewer, style, or focus
logic, then rerun the affected section and its automated tests. Sign off only
when every applicable check passes and no Level A or AA defect remains open.

Do not attach private learner questions, course content, screenshots, captured
answers, local paths, host details, or run records to a public issue. Follow the
[support-bundle guide](support-bundle.md) when maintainers need sanitized
diagnostic evidence.
