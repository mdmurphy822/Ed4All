/* Studio viewer shim — reanimates inert generated components inside the sandbox.
 *
 * Generated Courseforge pages ship as ZERO-JS self-contained HTML: the
 * self-check "Show Answer" buttons use inline `onclick="toggleAnswer(id)"` and
 * the flip-card grid relies on a `.flipped` class toggle. The page-serving path
 * (gui.services.imscc_service) STRIPS all <script> + inline on* handlers for
 * safety, so those interactions are dead in the served HTML. This shim — served
 * separately and injected into the sandboxed iframe by studio.js — re-binds them
 * with accessible, framework-free handlers. It runs with `allow-scripts` only
 * (NO allow-same-origin), so it cannot read the parent or make requests.
 *
 * Component contracts (from archived data-cf-* pages):
 *  - Reveal: <button class="reveal-btn" aria-controls="ID" aria-expanded="false">
 *    toggles the visibility of #ID (.answer-reveal, display:none by default).
 *  - Flip card: <div class="flip-card"> ... </div> toggles a `.flipped` class on
 *    click / Enter / Space; made focusable + role=button for keyboard parity.
 */
(function () {
  'use strict';

  function showReveal(target, btn) {
    target.style.display = 'block';
    btn.setAttribute('aria-expanded', 'true');
    if (/show/i.test(btn.textContent)) btn.textContent = btn.textContent.replace(/show/i, 'Hide');
  }
  function hideReveal(target, btn) {
    target.style.display = 'none';
    btn.setAttribute('aria-expanded', 'false');
    if (/hide/i.test(btn.textContent)) btn.textContent = btn.textContent.replace(/hide/i, 'Show');
  }

  function bindReveals() {
    var btns = document.querySelectorAll('.reveal-btn[aria-controls], button[aria-controls][aria-expanded]');
    Array.prototype.forEach.call(btns, function (btn) {
      var id = btn.getAttribute('aria-controls');
      var target = id && document.getElementById(id);
      if (!target) return;
      // Start collapsed (matches the page's default .answer-reveal display:none).
      if (btn.getAttribute('aria-expanded') !== 'true') target.style.display = 'none';
      btn.addEventListener('click', function () {
        var open = btn.getAttribute('aria-expanded') === 'true';
        if (open) hideReveal(target, btn); else showReveal(target, btn);
      });
    });
  }

  function bindFlipCards() {
    var cards = document.querySelectorAll('.flip-card');
    Array.prototype.forEach.call(cards, function (card) {
      if (!card.hasAttribute('tabindex')) card.setAttribute('tabindex', '0');
      if (!card.hasAttribute('role')) card.setAttribute('role', 'button');
      if (!card.hasAttribute('aria-pressed')) card.setAttribute('aria-pressed', 'false');
      function toggle() {
        var flipped = card.classList.toggle('flipped');
        card.setAttribute('aria-pressed', flipped ? 'true' : 'false');
      }
      card.addEventListener('click', toggle);
      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
          e.preventDefault();
          toggle();
        }
      });
    });
  }

  // Citation-back: scroll to the cited passage + temporarily highlight it.
  // studio.js stamps the target heading slug on <html data-cf-scroll-to="…">
  // when loading a page from a citation click (the parent can't reach into the
  // sandboxed iframe, so the hint rides the markup). The heading-id injector in
  // the page-serving path mints ids that match the citation heading slug.
  function scrollToCited() {
    var root = document.documentElement;
    var slug = root && root.getAttribute('data-cf-scroll-to');
    if (!slug) return;
    var target = document.getElementById(slug);
    if (!target) {
      // Fall back to a heading whose text slugifies to the same value.
      var hs = document.querySelectorAll('h1,h2,h3,h4,h5,h6');
      for (var i = 0; i < hs.length; i++) {
        var s = (hs[i].textContent || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
        if (s === slug) { target = hs[i]; break; }
      }
    }
    if (!target) return;
    try { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch (_) { target.scrollIntoView(); }
    target.classList.add('cf-cited-highlight');
    // Inject a minimal highlight style once (the served page strips author JS;
    // this inline style is scoped to the temporary class only).
    if (!document.getElementById('cf-cited-style')) {
      var st = document.createElement('style');
      st.id = 'cf-cited-style';
      st.textContent = '.cf-cited-highlight{background:#fff3bf;outline:2px solid #d9a400;transition:background .6s ease}';
      document.head.appendChild(st);
    }
    setTimeout(function () { target.classList.remove('cf-cited-highlight'); }, 3500);
  }

  function init() {
    try { bindReveals(); } catch (_) { /* ignore */ }
    try { bindFlipCards(); } catch (_) { /* ignore */ }
    try { scrollToCited(); } catch (_) { /* ignore */ }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
