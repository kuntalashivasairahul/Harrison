/* HarrisonGPT chat client.
   No dependencies.  The markdown renderer below handles only the subset the
   answer prompts can produce (### headings, **bold**, - bullets, [p:NNN]) and
   escapes everything before transforming it.  That is deliberate: model output
   is untrusted input at the render boundary, and a general markdown parser can
   emit arbitrary tags.  A whitelist is both smaller and safer here. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const form = $('ask-form'), input = $('q'), submit = $('submit');
  const pending = $('pending'), stages = $('stages'), errBox = $('error');
  const output = $('output'), doc = $('doc'), rail = $('rail');
  const confEl = $('conf'), echoEl = $('echo'), strip = $('strip');
  const lightbox = $('lightbox'), lbImg = $('lb-img');

  let mode = 'smart_summary';
  let inFlight = false;

  // ── rendering ───────────────────────────────────────────────────
  const esc = (s) => s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const CHIP = '<button type="button" class="cite" data-page="$1">p.$1</button>';

  // Runs on already-escaped text.  Order matters: citations last, so a page
  // label can never be swallowed by the bold pass.
  function inline(s) {
    return s
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // The model copies the marker format out of the retrieved context, which
      // carries a chunk id: [p:411|c:2197].  It sometimes emits the bare
      // [p:411] the prompt asks for.  Both forms have to render.
      // A trailing citation is also hoisted past the sentence period — the chip
      // carries its own padding, so "[p:3254]." leaves an orphaned full stop.
      .replace(/\s*\[p:(\d+)(?:\|c:\d+)?\]\s*\./g, '.' + CHIP)
      .replace(/\[p:(\d+)(?:\|c:\d+)?\]/g, CHIP);
  }

  function render(md) {
    const out = [];
    let list = null, quick = false;
    const flush = () => { if (list) { out.push('<ul>' + list.join('') + '</ul>'); list = null; } };

    for (const raw of md.split('\n')) {
      const line = raw.trim();
      if (!line) { flush(); continue; }

      const h = line.match(/^#{1,6}\s+(.*)$/);
      if (h) {
        flush();
        // The Quick Revision block closes a smart_summary; give it the well
        // treatment and keep everything after it inside.
        if (/^quick revision/i.test(h[1]) && !quick) { out.push('<div class="qr">'); quick = true; }
        out.push('<h3>' + inline(h[1]) + '</h3>');
        continue;
      }

      const li = line.match(/^[-*•]\s+(.*)$/);
      if (li) { (list ||= []).push('<li>' + inline(li[1]) + '</li>'); continue; }

      flush();
      out.push('<p>' + inline(line) + '</p>');
    }
    flush();
    if (quick) out.push('</div>');
    return out.join('');
  }

  // ── wait state ──────────────────────────────────────────────────
  // ponytail: estimated, not telemetry.  `timings` only arrives WITH the
  // answer, so mid-flight progress cannot be truthful.  Known ceiling: on a
  // slow request this parks on the last stage.  Real fix is an SSE endpoint.
  const PLAN = [['expanding', 900], ['retrieving', 700], ['reranking', 600],
                ['drafting', 9000], ['verifying', 6000]];
  let timers = [];

  function startStages() {
    stages.innerHTML = PLAN.map(([n]) => `<span class="st">${n}</span>`).join('');
    const nodes = [...stages.children];
    let at = 0, acc = 0;
    nodes[0].className = 'st now';
    PLAN.forEach(([, ms], i) => {
      if (i === PLAN.length - 1) return;
      acc += ms;
      timers.push(setTimeout(() => {
        nodes[at].className = 'st done';
        nodes[++at].className = 'st now';
      }, acc));
    });
  }
  const stopStages = () => { timers.forEach(clearTimeout); timers = []; };

  // ── citations ↔ page rail ───────────────────────────────────────
  function focusPage(page) {
    let hit = null;
    rail.querySelectorAll('.scan').forEach((s) => {
      const on = s.dataset.page === page;
      s.classList.toggle('active', on);
      if (on) hit = s;
    });
    if (hit) hit.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  doc.addEventListener('click', (e) => {
    const chip = e.target.closest('.cite');
    if (chip) focusPage(chip.dataset.page);
  });

  // Hovering a citation marks its page in the rail.  Same link as the click,
  // reversible half: no scroll, no pin, so a reader sweeping a paragraph can
  // see which scan each claim came from without committing to one.
  const peek = (page) => rail.querySelectorAll('.scan').forEach(
    (s) => s.classList.toggle('peek', s.dataset.page === page));

  doc.addEventListener('mouseover', (e) => {
    const chip = e.target.closest('.cite');
    if (chip && !chip.disabled) peek(chip.dataset.page);
  });
  doc.addEventListener('mouseout', (e) => {
    if (e.target.closest('.cite')) peek(null);
  });

  function openScan(url, label) {
    lbImg.src = url;
    lbImg.alt = 'Harrison’s page ' + label;
    lbImg.classList.remove('zoomed');   // each page opens fitted, not wherever
    lightbox.scrollTo(0, 0);            //   the last one was left panned
    lightbox.showModal();
  }

  // Fitted to a phone, a scanned page is legible only as a shape.  Tap to zoom,
  // then pan; the tapped point stays under the finger, because a zoom that
  // jumps to the middle of the page loses the paragraph you were aiming at.
  lbImg.addEventListener('click', (e) => {
    const r = lbImg.getBoundingClientRect();
    const nx = (e.clientX - r.left) / r.width;   // where on the page they tapped
    const ny = (e.clientY - r.top) / r.height;
    lbImg.classList.toggle('zoomed');
    // Read back after the class lands, so the scroll extent is the new one.
    requestAnimationFrame(() => {
      lightbox.scrollLeft = nx * lightbox.scrollWidth - e.clientX;
      lightbox.scrollTop = ny * lightbox.scrollHeight - e.clientY;
    });
  });
  $('lb-close').addEventListener('click', () => lightbox.close());
  lightbox.addEventListener('click', (e) => { if (e.target === lightbox) lightbox.close(); });

  // ── request ─────────────────────────────────────────────────────
  async function ask(query) {
    if (inFlight || !query.trim()) return;
    inFlight = true;
    submit.disabled = true;
    errBox.hidden = true;
    output.hidden = true;
    pending.hidden = false;
    startStages();

    let res, data;
    try {
      res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode }),
      });
      data = await res.json().catch(() => null);
    } catch {
      return fail('Could not reach the server. Check that it is running, then try again.');
    } finally {
      stopStages();
      pending.hidden = true;
      submit.disabled = false;
      inFlight = false;
    }

    if (!res.ok) {
      if (res.status === 429) return fail('Too many questions in a short window. Give it a moment.');
      if (res.status === 503) return fail('The service is degraded — the index or a model key is unavailable.');
      return fail((data && data.detail) || `Request failed (${res.status}).`);
    }
    show(query, data);
  }

  function fail(msg) {
    errBox.querySelector('p').textContent = msg;
    errBox.hidden = false;
  }

  const REFUSAL = 'Insufficient information in the provided context.';

  function show(query, r) {
    echoEl.textContent = query;

    const level = (r.confidence || 'Pending').toLowerCase();
    const dots = { high: '●●●', medium: '●●○', low: '●○○' }[level] || '';
    confEl.className = 'conf conf-' + (dots ? level : 'low');
    confEl.innerHTML = `<i>${dots}</i>${esc(r.confidence || 'Pending')} confidence`;

    // The refusal is a state of its own, rendered verbatim and unsoftened.
    if ((r.answer || '').trim() === REFUSAL) {
      doc.innerHTML =
        '<div class="refusal"><div class="k">No grounded answer</div>' +
        `<p>${esc(REFUSAL)}</p>` +
        '<p class="why">The retrieved passages did not support an answer, so none ' +
        'was written. Rephrasing may help; inventing a citation would not.</p></div>';
    } else {
      doc.innerHTML = render(r.answer || '');
    }

    // page rail
    const pages = r.visual_context || [];
    rail.innerHTML = pages.length
      ? '<div class="lbl">Cited pages</div>' + pages.map((p) => {
          const label = esc(p.page_label || '');
          const num = label.replace(/[^0-9]/g, '');
          return `<button type="button" class="scan" data-page="${num}" data-full="${esc(p.full_url)}">
            <img src="${esc(p.thumbnail_url)}" alt="Harrison's page ${label}" loading="lazy">
            <span class="cap"><span>${label}</span><span>&#8599;</span></span></button>`;
        }).join('')
      : '';
    rail.querySelectorAll('.scan').forEach((s) => {
      s.addEventListener('click', () => openScan(s.dataset.full, s.dataset.page));
    });

    // `sources` is derived from the retrieved chunks, not from the answer text,
    // so the model can cite a page that has no visual_context entry.  The
    // frontend must not build the URL itself — resolve_page_urls() is the only
    // permitted constructor (ARCHITECTURE §2) — so such a chip is made inert
    // rather than left as a click that silently does nothing.
    const have = new Set([...rail.querySelectorAll('.scan')].map((s) => s.dataset.page));
    doc.querySelectorAll('.cite').forEach((c) => {
      if (!have.has(c.dataset.page)) {
        c.disabled = true;
        c.classList.add('inert');
        c.title = 'Page image not among the retrieved sources';
      }
    });

    // pipeline strip — real numbers, straight from `timings`
    const t = r.timings || {};
    const order = [['optimizer', 'expand'], ['retrieval', 'retrieve'], ['reranking', 'rerank'],
                   ['draft_generation', 'draft'], ['verification', 'verify'], ['retry', 'retry']];
    const cells = order
      .filter(([k]) => t[k] > 0.0005)
      .map(([k, name]) => `<span><b>${name}</b> ${t[k].toFixed(2)}s</span>`);
    // Every stage reading zero means the semantic cache answered before the
    // pipeline ran.  Say so — otherwise the strip is one lonely "total" cell
    // and the number looks impossibly fast for no stated reason.
    if (!cells.length && t.total_request) cells.push('<span><b>served from cache</b></span>');
    if (t.total_request) cells.push(`<span><b>total</b> ${t.total_request.toFixed(2)}s</span>`);
    strip.innerHTML = cells.join('');
    strip.hidden = !cells.length;

    output.hidden = false;
    output.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  // ── wiring ──────────────────────────────────────────────────────
  form.addEventListener('submit', (e) => { e.preventDefault(); ask(input.value); });

  document.querySelectorAll('.seg button').forEach((b) => {
    b.addEventListener('click', () => {
      mode = b.dataset.mode;
      document.querySelectorAll('.seg button').forEach((o) =>
        o.setAttribute('aria-pressed', String(o === b)));
    });
  });

  document.querySelectorAll('.ex').forEach((b) => {
    b.addEventListener('click', () => { input.value = b.textContent.trim(); ask(input.value); });
  });
})();
