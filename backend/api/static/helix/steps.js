/* Which scroll step is active, and the DOM that follows from it.
 *
 * Reads the *undamped* progress.  The helix uses the damped value because a
 * wheel notch rotated straight through looks like a stutter; the copy must not,
 * because a reader who scrolls to step 3 and sees step 2's headline for another
 * half second reads that as the page being broken.
 *
 * Nothing here creates or destroys elements — every step and card is already in
 * the DOM (that is what makes the copy readable with JS off).  This only
 * toggles data-active, and the stylesheet owns the crossfade.
 */

/* (spec) The band a boundary has to be crossed by before the step commits.
 * Without it a reader parked exactly on a boundary — which a trackpad makes
 * easy — flickers between two headlines on every stray pixel of scroll. */
const HYSTERESIS = 0.04;

/**
 * @param {HTMLElement} hero
 * @returns {{update(raw:number):void, index:number}|null} null when this
 *          machinery must not run at all
 */
export function createSteps(hero) {
  // Under reduced motion the stylesheet unpins the section and puts every step
  // in normal flow.  Managing data-active then would hide four of five steps
  // from assistive tech while they are plainly visible on screen.
  if (matchMedia('(prefers-reduced-motion:reduce)').matches) return null;

  const steps = [...hero.querySelectorAll('.helix-step')];
  const cards = [...hero.querySelectorAll('.helix-card')];
  const rail = [...hero.querySelectorAll('.helix-hero__rail a')];
  const n = steps.length;
  if (!n) return null;

  let index = -1;

  function apply(i) {
    if (i === index) return;
    index = i;
    // The CSS matches [data-active="true"], so this cannot use toggleAttribute:
    // that sets the value to the empty string and the selector never fires.
    const mark = (el, on) => {
      if (on) el.setAttribute('data-active', 'true'); else el.removeAttribute('data-active');
      // visibility:hidden already takes inactive steps out of the accessibility
      // tree and the tab order; this states the intent rather than relying on a
      // stylesheet the markup cannot see.
      if (on) el.removeAttribute('aria-hidden'); else el.setAttribute('aria-hidden', 'true');
    };
    steps.forEach((el, k) => mark(el, k === i));
    cards.forEach((el, k) => mark(el, k === i));
    // The rail is indexed by position, not by rail_key: HeroStep.rail_key and
    // RAIL are declared in the same Python file and kept in the same order.
    rail.forEach((a, k) => {
      if (k === i) a.setAttribute('aria-current', 'step'); else a.removeAttribute('aria-current');
    });
  }

  /* Adjacent moves step one index at a time, because the hysteresis band is
   * defined against a single boundary and there is nothing to evaluate a
   * two-boundary jump against.  Non-adjacent moves snap instead: walking them
   * one per frame makes every intermediate step visible for a frame, and since
   * the crossfade out takes .55s the reader sees three headlines stacked on a
   * fast flick.  A gap that large is also unambiguous — nobody lands two steps
   * away by hovering on a boundary, so there is nothing for the band to
   * protect against. */
  function update(raw) {
    const want = Math.min(n - 1, Math.floor(raw * n));
    if (want === index) return;
    if (index < 0 || Math.abs(want - index) > 1) { apply(want); return; }
    if (want > index) {
      if (raw > (index + 1) / n + HYSTERESIS) apply(index + 1);
    } else if (raw < index / n - HYSTERESIS) {
      apply(index - 1);
    }
  }

  /* The rail's href points at #step-N, but every step sits in the same pinned
   * cell — the browser would scroll to the section and stop, landing on step 1
   * whichever link was clicked.  Translate the target into the scroll offset
   * that makes that step active instead. */
  rail.forEach((a, k) => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      const r = hero.getBoundingClientRect();
      const span = Math.max(1, r.height - innerHeight);
      // Mid-band, so the landing position is not sitting on a boundary.
      scrollTo({ top: r.top + scrollY + span * ((k + 0.5) / n), behavior: 'smooth' });
    });
  });

  apply(0);
  return { update, get index() { return index; } };
}
