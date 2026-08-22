/* Scroll position for the pinned section, as a 0..1 number.
 *
 * Two values, deliberately:
 *
 *   raw     — where the scrollbar actually is, updated synchronously on the
 *             scroll event.  Step switching reads this, because a damped
 *             value lags and the copy would change after the helix has
 *             already moved on.
 *   damped  — raw chased at a fixed rate, advanced once per frame.  The helix
 *             reads this, because a wheel notch is a step function and
 *             rotating by it directly looks like a stutter, not a scroll.
 *
 * The scroll handler does arithmetic only.  getBoundingClientRect() in a
 * scroll handler forces a synchronous layout on every event, which is exactly
 * the cost this file exists to avoid — the geometry is measured on resize and
 * cached.
 */
const clamp01 = (n) => (n < 0 ? 0 : n > 1 ? 1 : n);

/**
 * @param {HTMLElement} section  the tall wrapper, not the sticky child
 * @param {number} damping       (spec) 0.085 — fraction of the gap closed per frame
 */
export function createScrollProgress(section, damping = 0.085) {
  let top = 0;
  let span = 1;
  let raw = 0;
  let damped = 0;

  function measure() {
    const r = section.getBoundingClientRect();
    top = r.top + scrollY;
    // The pin is released once the section's bottom reaches the viewport
    // bottom, so the travel is its height minus one viewport — not its height.
    // max(1, ...) keeps this out of a divide by zero when the section is
    // shorter than the viewport, which is what the reduced-motion and no-JS
    // rules make it.
    span = Math.max(1, r.height - innerHeight);
    read();
  }

  function read() {
    raw = clamp01((scrollY - top) / span);
  }

  measure();
  addEventListener('scroll', read, { passive: true });
  addEventListener('resize', measure, { passive: true });

  return {
    get raw() { return raw; },
    get damped() { return damped; },
    /** Advance the damped value one frame and return it. */
    step() {
      damped += (raw - damped) * damping;
      return damped;
    },
    /** Snap both values together — used when the loop resumes after a pause,
     *  so the helix does not spin through the interval it slept for. */
    settle() {
      damped = raw;
    },
    measure,
  };
}
