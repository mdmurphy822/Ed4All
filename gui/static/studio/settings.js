/* Ed4All Studio — simplified Settings page (Marketable-v1 C3).
 *
 * The credentials / answer / global SUBSET of the env catalog relevant to a
 * Studio user: the AI provider + model, the grounded-answer (ask) backend, and
 * cloud-provider API keys (masked). GUI host/port are shown read-only. Writes
 * go through the existing settings API (PATCH /api/settings deep-merge of dotted
 * paths); secrets are written under ``env`` and the page never echoes them back.
 * A "Test provider" button hits POST /api/settings/test-provider for a cheap
 * reachability check (no model generation) and reports ok / unreachable /
 * unauthorized clearly.
 *
 * Renders entirely from /api/settings/studio (the scoped payload) so it never
 * hardcodes the catalog.
 */

import { api, apiJSON } from '/shared/api.js';
import { el, clear, uid } from '/shared/dom.js';
import { toast, toastErr } from '/shared/toast.js';

export async function renderSettings(shell) {
  shell.crumbs([{ text: 'Library', href: '#/library' }, { text: 'Settings' }]);
  const v = clear(shell.view());
  shell.setBusy(true);
  shell.setStatus('Loading settings.');

  v.appendChild(el('h1', { text: 'Settings' }));

  let doc;
  try {
    doc = await api('/api/settings/studio');
  } catch (e) {
    shell.setBusy(false);
    v.appendChild(el('p', { class: 'error', role: 'alert', text: shell.errText(e) }));
    toastErr(e, 'Settings load failed');
    return;
  }
  shell.setBusy(false);

  const providers = doc.providers || [];
  const routing = doc.model_routing || {};
  const env = doc.env || {};
  const catalog = doc.catalog || [];

  const form = el('form', { class: 'settings-form', novalidate: true });

  // ---- AI provider (global model_routing) ---------------------------------
  const g = routing.global || {};
  form.appendChild(el('h2', { text: 'AI provider' }));

  const modeId = uid('mode');
  form.appendChild(el('div', { class: 'field' }, [
    el('label', { for: modeId, text: 'Mode' }),
    selectEl(modeId, ['local', 'api'], g.mode || 'local'),
    el('p', { class: 'field-hint', text: 'local: run on this machine (no key needed). api: call a cloud provider.' }),
  ]));
  const modeSel = form.querySelector(`#${CSS.escape(modeId)}`);

  const provId = uid('prov');
  form.appendChild(el('div', { class: 'field' }, [
    el('label', { for: provId, text: 'Provider' }),
    providerSelect(provId, providers, g.provider || 'local'),
  ]));
  const provSel = form.querySelector(`#${CSS.escape(provId)}`);

  const modelId = uid('model');
  form.appendChild(el('div', { class: 'field' }, [
    el('label', { for: modelId, text: 'Model (optional)' }),
    el('input', { id: modelId, type: 'text', value: g.model || '', autocomplete: 'off', placeholder: 'provider default' }),
    el('p', { class: 'field-hint', text: 'Leave blank to use the provider’s default model.' }),
  ]));
  const modelInput = form.querySelector(`#${CSS.escape(modelId)}`);

  // Test provider button + result line.
  const testResult = el('p', { class: 'test-result', role: 'status', 'aria-live': 'polite' });
  const testBtn = el('button', { type: 'button', class: 'btn', text: 'Test provider' });
  form.appendChild(el('div', { class: 'field' }, [testBtn, testResult]));
  testBtn.addEventListener('click', () => testProvider(provSel.value, testBtn, testResult, shell));

  // ---- Answer (ask) backend ----------------------------------------------
  const a = routing.answer || {};
  form.appendChild(el('h2', { text: 'Answers (course Q&A)' }));
  form.appendChild(el('p', { class: 'field-hint', text: 'The local backend that answers learner questions. Loopback-only.' }));
  const ansModelId = uid('ans-model');
  form.appendChild(el('div', { class: 'field' }, [
    el('label', { for: ansModelId, text: 'Answer model (optional)' }),
    el('input', { id: ansModelId, type: 'text', value: a.model || '', autocomplete: 'off', placeholder: 'local default' }),
  ]));
  const ansModelInput = form.querySelector(`#${CSS.escape(ansModelId)}`);

  // ---- API keys (credentials, masked) ------------------------------------
  const credKeys = catalog.filter((c) => c.category === 'credentials');
  if (credKeys.length) {
    form.appendChild(el('h2', { text: 'API keys' }));
    form.appendChild(el('p', { class: 'field-hint', text: 'Only needed for cloud providers. Stored masked; leave blank to keep the saved value.' }));
  }
  const keyInputs = new Map();
  credKeys.forEach((c) => {
    const id = uid('key');
    const isSet = env[c.key] === 'set';
    const field = el('div', { class: 'field' }, [
      el('label', { for: id, text: c.label }),
      el('input', {
        id,
        type: 'password',
        autocomplete: 'off',
        'aria-describedby': `${id}-h`,
        placeholder: isSet ? '•••••• (saved)' : 'not set',
      }),
      el('p', { id: `${id}-h`, class: 'field-hint', text: c.help || '' }),
    ]);
    form.appendChild(field);
    keyInputs.set(c.key, field.querySelector('input'));
  });

  // ---- Read-only host/port -----------------------------------------------
  form.appendChild(el('h2', { text: 'Server' }));
  form.appendChild(el('div', { class: 'field readonly' }, [
    el('span', { class: 'ro-label', text: 'Address' }),
    el('span', { class: 'ro-value', text: `${doc.host || '127.0.0.1'}:${doc.port || 8077}` }),
    el('p', { class: 'field-hint', text: 'Where Studio is served. Change this from the command line, not the browser.' }),
  ]));

  // ---- Save ---------------------------------------------------------------
  const saveBtn = el('button', { type: 'submit', class: 'btn primary', text: 'Save settings' });
  form.appendChild(el('div', { class: 'wizard-nav' }, [
    el('a', { class: 'btn', href: '#/library', text: 'Back to library' }),
    saveBtn,
  ]));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    saveBtn.disabled = true;
    const patch = {
      model_routing: {
        global: { mode: modeSel.value, provider: provSel.value, model: modelInput.value.trim() || null },
        answer: { provider: a.provider || null, model: ansModelInput.value.trim() || null },
      },
    };
    // Only include keys the user actually typed (blank = keep saved value).
    const envPatch = {};
    keyInputs.forEach((input, key) => {
      const val = input.value;
      if (val && val.trim()) envPatch[key] = val.trim();
    });
    if (Object.keys(envPatch).length) patch.env = envPatch;

    try {
      await apiJSON('/api/settings', 'PATCH', patch);
      toast('Settings saved.', '', 'success');
      shell.setStatus('Settings saved.');
      // Clear the secret inputs so a typed key doesn't linger in the DOM.
      keyInputs.forEach((input) => { input.value = ''; });
    } catch (err) {
      toastErr(err, 'Save failed');
      shell.setStatus('Could not save settings.');
    } finally {
      saveBtn.disabled = false;
    }
  });

  v.appendChild(form);
}

/* --------------------------------------------------------------- test btn */

async function testProvider(provider, btn, resultEl, shell) {
  btn.disabled = true;
  resultEl.className = 'test-result';
  resultEl.textContent = 'Testing…';
  shell.setStatus(`Testing the ${provider} provider.`);
  let res;
  try {
    res = await apiJSON('/api/settings/test-provider', 'POST', { provider });
  } catch (e) {
    resultEl.className = 'test-result is-fail';
    resultEl.textContent = `Test failed: ${shell.errText(e)}`;
    shell.setStatus('Provider test failed.');
    btn.disabled = false;
    return;
  }
  // Map the service's status enum to a clear, end-user-facing message.
  const status = res.status || (res.ok ? 'ok' : 'failed');
  const human = humanizeTest(res);
  resultEl.className = `test-result ${res.ok ? 'is-ok' : 'is-fail'}`;
  resultEl.textContent = human;
  shell.setStatus(human);
  btn.disabled = false;
}

/** Turn the {ok,status,detail} probe result into one plain-language line. */
function humanizeTest(res) {
  const detail = res.detail ? ` ${res.detail}` : '';
  switch (res.status) {
    case 'reachable':
      return `Connected. The provider is reachable.${detail}`;
    case 'ready':
      return `Ready.${detail}`;
    case 'key_present':
      return `API key is set (no live call made).${detail}`;
    case 'missing_key':
      return `Not authorized: no API key set for this provider.${detail}`;
    case 'unreachable':
      return `Unreachable: could not connect to the provider.${detail}`;
    case 'http_error':
      return `Reachable but returned an error response.${detail}`;
    case 'ping_failed':
      return `Could not authenticate / reach the provider.${detail}`;
    case 'unknown_provider':
      return `Unknown provider.${detail}`;
    default:
      return res.ok ? `OK.${detail}` : `Test failed.${detail}`;
  }
}

/* --------------------------------------------------------------- helpers */

function selectEl(id, options, current) {
  const sel = el('select', { id });
  options.forEach((opt) => {
    sel.appendChild(el('option', { value: opt, text: opt, selected: opt === current ? true : null }));
  });
  return sel;
}

function providerSelect(id, providers, current) {
  const sel = el('select', { id });
  providers.forEach((p) => {
    sel.appendChild(el('option', { value: p.name, text: p.label || p.name, selected: p.name === current ? true : null }));
  });
  return sel;
}
