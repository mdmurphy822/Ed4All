/* fieldRow — a label-paired input row. Vanilla ES module, NO build.
 *
 * The future jargon-fix primitive: renders a {label, help} pair bound to a
 * control via <label for> + aria-describedby, so the Operator Settings surface
 * can show env_catalog.entry.label / .help instead of a bare env-var name.
 * PURE PRESENTATION: takes data as args, calls no API.
 *
 * a11y: every control gets a real <label for=id>; the help text is wired via
 * aria-describedby; an optional error is a role="alert" region (hidden until
 * populated). Mirrors the studio wizard field pattern the WCAG gate validates.
 */

import { el, uid } from '../dom.js';

/**
 * Build a labelled field row.
 * @param {Object} field
 * @param {string} field.label            human label (e.g. "Outline course model").
 * @param {string} [field.help]           help / description text.
 * @param {string} [field.type="text"]    input type (text|password|number|search|…)
 *                                         or "select" / "textarea".
 * @param {string} [field.value=""]       initial value.
 * @param {string} [field.placeholder]
 * @param {Array<{value,label,selected}>} [field.options]  for type="select".
 * @param {string} [field.name]           form field name (and data attr).
 * @param {boolean} [field.required]
 * @returns {{el: HTMLElement, control: HTMLElement}}
 */
export function fieldRow(field = {}) {
  const id = uid('fr');
  const type = field.type || 'text';
  const helpId = field.help ? `${id}-h` : null;

  let control;
  if (type === 'select') {
    control = el('select', { id, name: field.name, 'aria-describedby': helpId });
    (field.options || []).forEach((o) => {
      control.appendChild(el('option', { value: o.value, selected: !!o.selected, text: o.label != null ? o.label : o.value }));
    });
  } else if (type === 'textarea') {
    control = el('textarea', { id, name: field.name, rows: field.rows || 3, 'aria-describedby': helpId, placeholder: field.placeholder });
    if (field.value) control.value = field.value;
  } else {
    control = el('input', {
      id, name: field.name, type,
      value: field.value || '',
      placeholder: field.placeholder,
      'aria-describedby': helpId,
      required: field.required ? true : false,
      autocomplete: type === 'password' ? 'off' : (field.autocomplete || null),
    });
  }

  const label = el('label', { class: 'field-label', text: field.label || field.name || 'Field' });
  label.htmlFor = id;

  const kids = [
    label,
    control,
    field.help ? el('p', { id: helpId, class: 'field-hint', text: field.help }) : null,
    el('p', { class: 'error field-error', role: 'alert', hidden: true }),
  ];
  return { el: el('div', { class: 'field field-row' }, kids), control };
}

export default fieldRow;
