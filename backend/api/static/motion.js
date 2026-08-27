/* HarrisonGPT — pointer motion.  Pure progressive enhancement: the whole file
   no-ops on a coarse pointer or under prefers-reduced-motion, and every effect
   is published as a CSS custom property, so the stylesheet decides how far each
   one travels.  One pointermove listener, one rAF, no dependencies.

   Four effects, all of them the "reading lamp over a printed page" metaphor the
   design system is already built on:
     lamp    — the paper ground warms around the cursor
     tilt    — cards lean a couple of degrees toward it, like handled paper
     magnet  — buttons lean the same way, a few px
     loupe   — a cited page magnifies under the cursor, read without opening it */
(() => {
  'use strict';

  // hover:hover as well as pointer:fine — an iPad with an Apple Pencil reports
  // a fine pointer but cannot hover, and every effect here is built on the
  // pointer resting somewhere.  Without this the loupe opens on tap and stays.
  if (!matchMedia('(hover:hover) and (pointer:fine)').matches) return;
  if (matchMedia('(prefers-reduced-motion:reduce)').matches) return;

  const root = document.documentElement;
  root.classList.add('motion');

  const CARD = '.close,.demo .ans,.answer,.scan';
  const MAGNET = 4;    // px a button leans toward the pointer
  const LOUPE = 168;   // px diameter, must match .loupe in site.css
  const ZOOM = 2.6;

  let px = 0, py = 0, queued = false;
  let card = null, cardRect = null;
  let btn = null, btnRect = null;
  let loupe = null, target = null, targetRect = null;  // target: the <img> magnified

  // ── the single frame ────────────────────────────────────────────
  // Rects are read on enter, never per frame: reading one inside the rAF forces
  // a layout on every move, and none of these elements shift while hovered.
  function frame() {
    queued = false;
    root.style.setProperty('--px', px + 'px');
    root.style.setProperty('--py', py + 'px');

    if (card) {
      const r = cardRect;
      const nx = (px - r.left) / r.width, ny = (py - r.top) / r.height;
      card.style.setProperty('--tx', (nx - 0.5).toFixed(3));
      card.style.setProperty('--ty', (ny - 0.5).toFixed(3));
      card.style.setProperty('--lx', (nx * 100).toFixed(1) + '%');
      card.style.setProperty('--ly', (ny * 100).toFixed(1) + '%');
    }

    if (btn) {
      const r = btnRect;
      btn.style.setProperty('--mx', (((px - r.left) / r.width - 0.5) * 2 * MAGNET).toFixed(2));
      btn.style.setProperty('--my', (((py - r.top) / r.height - 0.5) * 2 * MAGNET).toFixed(2));
    }

    if (target) placeLoupe();
  }

  addEventListener('pointermove', (e) => {
    px = e.clientX; py = e.clientY;
    if (!queued) { queued = true; requestAnimationFrame(frame); }
  }, { passive: true });

  // ── hover tracking ──────────────────────────────────────────────
  addEventListener('pointerover', (e) => {
    const c = e.target.closest(CARD);
    if (c !== card) { dropCard(); card = c; cardRect = c && c.getBoundingClientRect(); }

    const b = e.target.closest('.btn');
    if (b !== btn) { dropBtn(); btn = b; btnRect = b && b.getBoundingClientRect(); }

    // Only a rail scan carries data-full — the landing page's facsimile is a CSS
    // pattern with nothing behind it to magnify.
    const scan = e.target.closest('.scan[data-full]');
    if (scan) raiseLoupe(scan); else dropLoupe();
  }, { passive: true });

  addEventListener('pointerleave', () => { dropCard(); dropBtn(); dropLoupe(); });

  // The page scrolls under a still cursor — a sticky rail, or chat.js scrolling
  // a fresh answer into view — and every cached rect becomes a lie.  Re-measure
  // rather than drop: dropping looks like a fix but leaves the effect dead until
  // the pointer re-enters the element, which is exactly what it will not do
  // while the reader sits still and reads.
  addEventListener('scroll', () => {
    if (!card && !btn && !target) return;
    if (card) cardRect = card.getBoundingClientRect();
    if (btn) btnRect = btn.getBoundingClientRect();
    if (target) targetRect = target.getBoundingClientRect();
    if (!queued) { queued = true; requestAnimationFrame(frame); }
  }, { passive: true });

  function dropCard() {
    if (card) ['--tx', '--ty', '--lx', '--ly'].forEach((p) => card.style.removeProperty(p));
    card = cardRect = null;
  }
  function dropBtn() {
    if (btn) { btn.style.removeProperty('--mx'); btn.style.removeProperty('--my'); }
    btn = btnRect = null;
  }

  // ── loupe ───────────────────────────────────────────────────────
  function raiseLoupe(scan) {
    const img = scan.querySelector('img');
    if (!img) return;
    if (!loupe) {
      loupe = document.createElement('div');
      loupe.className = 'loupe';
      loupe.setAttribute('aria-hidden', 'true');
      document.body.appendChild(loupe);
    }
    const url = scan.dataset.full;
    if (loupe.dataset.src !== url) {
      loupe.dataset.src = url;
      loupe.style.backgroundImage = `url("${url.replace(/"/g, '%22')}")`;
    }
    target = img;
    targetRect = img.getBoundingClientRect();
    placeLoupe();
  }

  function placeLoupe() {
    const r = targetRect;
    const nx = (px - r.left) / r.width, ny = (py - r.top) / r.height;
    if (nx < 0 || nx > 1 || ny < 0 || ny > 1) { loupe.classList.remove('on'); return; }
    // The full scan and its thumbnail share an aspect ratio, so sizing the
    // backdrop off the thumbnail's own box keeps the magnification honest.
    const bw = r.width * ZOOM, bh = r.height * ZOOM;
    loupe.style.backgroundSize = `${bw}px ${bh}px`;
    loupe.style.backgroundPosition = `${LOUPE / 2 - nx * bw}px ${LOUPE / 2 - ny * bh}px`;
    loupe.style.left = px + 'px';
    loupe.style.top = py + 'px';
    loupe.classList.add('on');
  }

  function dropLoupe() {
    if (loupe) loupe.classList.remove('on');
    target = targetRect = null;
  }
})();
