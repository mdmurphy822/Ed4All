/* Shared hash router — vanilla ES module, NO build step.
 *
 * A tiny location.hash router. Routes are matched on the FIRST path segment
 * after `#/` (e.g. `#/viewer/course-alpha` → route "viewer", rest ["course-alpha"]).
 * The handler receives the trailing segments + the raw hash so a viewer route
 * can read its course slug / item without a framework.
 *
 * Generalized from gui/static/app.js's currentTab()/route() so the operator SPA
 * can later adopt it; the Studio surface uses it now.
 */

/**
 * Create a router.
 * @param {Object} routes  map of name -> async (segments, rawHash) => teardown?
 * @param {Object} opts    { defaultRoute, onError(err) }
 * @returns {{ start: () => void, navigate: (hash) => void, current: () => string }}
 */
export function createRouter(routes, opts = {}) {
  const defaultRoute = opts.defaultRoute || Object.keys(routes)[0];
  let activeTeardown = null;

  function parse() {
    const raw = (location.hash || '').replace(/^#\/?/, '');
    const segs = raw.split('/').filter(Boolean).map(decodeURIComponent);
    const name = segs[0] && routes[segs[0]] ? segs[0] : defaultRoute;
    return { name, segments: segs.slice(1), raw };
  }

  async function dispatch() {
    if (typeof activeTeardown === 'function') {
      try { activeTeardown(); } catch (_) { /* ignore */ }
      activeTeardown = null;
    }
    const { name, segments, raw } = parse();
    try {
      const teardown = await routes[name](segments, raw);
      if (typeof teardown === 'function') activeTeardown = teardown;
    } catch (err) {
      if (typeof opts.onError === 'function') opts.onError(err);
      else throw err;
    }
  }

  return {
    start() {
      window.addEventListener('hashchange', dispatch);
      if (!location.hash) location.hash = `#/${defaultRoute}`;
      else dispatch();
    },
    navigate(hash) {
      const next = hash.startsWith('#') ? hash : `#/${hash}`;
      if (location.hash === next) dispatch();
      else location.hash = next;
    },
    current() { return parse().name; },
  };
}
