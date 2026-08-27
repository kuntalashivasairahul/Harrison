/* Self-check for the answer renderer in backend/api/static/chat.js.
 *
 * That renderer escapes model output and then applies four whitelisted
 * transforms.  Model output is untrusted input at the render boundary, so the
 * property that matters is: no matter what the model emits, the constructed
 * DOM contains only <p> <h3> <ul> <li> <strong> <button> <div>.
 *
 * Named probe_* like the Python diagnostics in this directory — it is run by
 * hand, and pytest never collects it.  Node is not a project dependency; this
 * is the one place it is used.
 *
 *   node scripts/probe_chat_render.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const src = fs.readFileSync(path.join(root, 'backend/api/static/chat.js'), 'utf8');

// Lift the pure functions out of the browser IIFE.
const CHIP = '<button type="button" class="cite" data-page="$1">p.$1</button>';
const esc = eval('(' + src.match(/const esc = ([\s\S]*?);\n\n/)[1] + ')');
const inline = eval('(' + src.match(/function inline\(s\) \{[\s\S]*?\n {2}\}/)[0] + ')');
const render = eval('(' + src.match(/function render\(md\) \{[\s\S]*?\n {2}\}/)[0] + ')');

const ALLOWED = new Set(['P', 'H3', 'UL', 'LI', 'STRONG', 'BUTTON', 'DIV']);
let failures = 0;

const check = (name, cond, detail = '') => {
  if (!cond) { failures++; console.log(`FAIL  ${name}  ${detail}`); }
  else console.log(`ok    ${name}`);
};

/* 1. Hostile model output must never construct a disallowed element. */
const hostile = [
  '<img src=x onerror=alert(1)>',
  '<script>alert(1)</script>',
  '<a href="javascript:alert(1)">c</a>',
  '<svg onload=alert(1)>',
  '"><button onclick=alert(1)>x',
  '### <img src=x onerror=alert(1)>',
  '- <iframe src=//evil></iframe>',
  '**<style>body{display:none}</style>**',
];
for (const probe of hostile) {
  const html = render(esc(probe));
  const tags = [...html.matchAll(/<\/?([a-zA-Z][a-zA-Z0-9]*)/g)].map((m) => m[1].toUpperCase());
  const bad = [...new Set(tags.filter((t) => !ALLOWED.has(t)))];
  check(`escapes ${JSON.stringify(probe.slice(0, 34))}`, bad.length === 0, `built <${bad}>`);
}

/* 2. Both citation marker shapes must render.  The model copies the format out
 *    of the retrieved context, which carries a chunk id. */
check('renders [p:411|c:2197]', render(esc('Fluids [p:411|c:2197]')).includes('data-page="411"'));
check('renders bare [p:411]', render(esc('Fluids [p:411]')).includes('data-page="411"'));
check('leaves no raw marker', !render(esc('a [p:411|c:2197] b [p:9]')).includes('[p:'));

/* 3. A trailing citation hoists past the sentence period, or the chip's own
 *    padding strands the full stop. */
const hoisted = render(esc('Give saline [p:3254].'));
check('hoists citation past period', /saline\.<button/.test(hoisted), hoisted);

/* 4. Structural transforms. */
check('### becomes h3', render(esc('### Fluid Replacement')).includes('<h3>Fluid Replacement</h3>'));
check('- becomes li', render(esc('- one\n- two')).includes('<li>one</li><li>two</li>'));
check('**x** becomes strong', render(esc('**bold**')).includes('<strong>bold</strong>'));
check('Quick Revision opens a well', render(esc('### Quick Revision\n- a')).includes('class="qr"'));
check('Quick Revision closes', (() => {
  const h = render(esc('### Quick Revision\n- a'));
  return (h.match(/<div/g) || []).length === (h.match(/<\/div>/g) || []).length;
})());

/* 5. The refusal string is duplicated in chat.js and backend/llm/llm.py.  If the
 *    two drift, the refusal silently renders as an ordinary paragraph — no
 *    error, no visible symptom, just the load-bearing state quietly gone. */
const jsRefusal = src.match(/const REFUSAL = '([^']+)'/)[1];
const pyRefusal = fs.readFileSync(path.join(root, 'backend/llm/llm.py'), 'utf8')
  .match(/^REFUSAL_STR = "([^"]+)"/m)[1];
check('refusal string matches backend', jsRefusal === pyRefusal,
      `js=${JSON.stringify(jsRefusal)} py=${JSON.stringify(pyRefusal)}`);

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
