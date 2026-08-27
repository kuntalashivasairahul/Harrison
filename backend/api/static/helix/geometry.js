/* Point cloud for the double helix.
 *
 * Built once, on the CPU, into flat typed arrays.  Nothing here animates —
 * the whole cloud is a static THREE.Points that the controller rotates.  That
 * is the reason 35k particles cost nothing per frame: no per-particle work
 * happens after this file returns.
 *
 * Constants marked (spec) are the verified values from HELIX_HERO_SPEC §7 and
 * must not be re-derived.  The rest are this site's staging — how tall the
 * helix is, where it sits — and are safe to tune.
 */
import * as THREE from 'three';

export const COUNT_FULL = 35140;      // (spec) particles at quality 1

// Staging.  HEIGHT is deliberately larger than the visible frustum at the
// camera's near position: the helix has to bleed off the top edge and dissolve
// at the bottom rather than sit inside a box.
// HEIGHT is a compromise with the gradient, not a free choice.  The camera
// sees ~18.6 world units of height, so a taller helix bleeds off frame more
// convincingly but shows a narrower slice of the colour ramp — at HEIGHT 34
// only stops 1..4 were ever on screen and the whole thing read as one crimson.
// 24 still overflows the frustum at both ends while putting nearly the full
// ramp in view.  TURNS follows to keep the pitch at 10 units per turn.
const TURNS = 2.4;
const HEIGHT = 24;
const RADIUS = 4.2;
const RUNGS = 88;

// Fractions of the budget.  Backbones carry the readable shape, rungs make it
// legible as a *double* helix rather than two unrelated spirals, dust keeps the
// silhouette from ending on a hard edge.
const F_BACKBONE = 0.5008;
const F_RUNGS = 0.3255;

/* Box–Muller.  (spec)  Uniform noise gives the cloud a visible boxy edge;
 * a gaussian falls off smoothly, which is what makes the strands read as glow
 * rather than as scatter.  1 - random() rather than random(): Math.random()
 * can return exactly 0 and log(0) is -Infinity, which would put a particle at
 * NaN and silently blank the entire buffer. */
function gauss() {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* Colour along the helix.  `stops` are CSS colour strings read from the design
 * tokens, so the palette lives in tokens.css and this file never hardcodes one.
 *
 * The (t - 0.05) / 0.72 remap is (spec): it pushes the gradient's full range
 * into the band that is actually on screen.  Without it the end stops land in
 * the parts of the helix that bleed off frame and the visible section is all
 * middle colours.
 */
function gradient(stops) {
  const cols = stops.map((s) => new THREE.Color().setStyle(s, THREE.SRGBColorSpace));
  const seg = cols.length - 1;
  return (t, out) => {
    const u = Math.min(1, Math.max(0, (t - 0.05) / 0.72));
    const f = u * seg;
    const i = Math.min(seg - 1, Math.floor(f));
    out.copy(cols[i]).lerp(cols[i + 1], f - i);
    return out;
  };
}

/**
 * @param {string[]} stops  five CSS colours, dark end first
 * @param {number} quality  1 = full budget; lower tiers scale the count
 * @returns {THREE.BufferGeometry} with position, aColor and aSize attributes
 */
export function buildHelix(stops, quality = 1) {
  const total = Math.round(COUNT_FULL * quality);
  const nBack = Math.round(total * F_BACKBONE) & ~1;   // even: split across 2 strands
  const nRung = Math.round(total * F_RUNGS);
  const nDust = total - nBack - nRung;

  const pos = new Float32Array(total * 3);
  const col = new Float32Array(total * 3);
  const siz = new Float32Array(total);
  const c = new THREE.Color();
  const colourAt = gradient(stops);
  let w = 0;

  const write = (x, y, z, size) => {
    pos[w * 3] = x; pos[w * 3 + 1] = y; pos[w * 3 + 2] = z;
    colourAt((y + HEIGHT / 2) / HEIGHT, c);
    col[w * 3] = c.r; col[w * 3 + 1] = c.g; col[w * 3 + 2] = c.b;
    siz[w] = size;
    w++;
  };

  // ── the two backbones ──────────────────────────────────────────────
  const perStrand = nBack / 2;
  for (let strand = 0; strand < 2; strand++) {
    const phase = strand * Math.PI;        // the second strand is half a turn behind
    for (let i = 0; i < perStrand; i++) {
      const t = i / perStrand;
      const a = t * TURNS * Math.PI * 2 + phase;
      // Jitter is applied in the strand's own frame, so the tube thickness is
      // constant along its length instead of flaring where the curve is steep.
      const r = RADIUS + gauss() * 0.22;
      write(
        Math.cos(a) * r + gauss() * 0.06,
        (t - 0.5) * HEIGHT + gauss() * 0.1,
        Math.sin(a) * r + gauss() * 0.06,
        0.85 + Math.random() * 0.5,
      );
    }
  }

  // ── base-pair rungs ────────────────────────────────────────────────
  const perRung = Math.floor(nRung / RUNGS);
  for (let k = 0; k < RUNGS; k++) {
    const t = (k + 0.5) / RUNGS;
    const a = t * TURNS * Math.PI * 2;
    const y = (t - 0.5) * HEIGHT;
    const x0 = Math.cos(a) * RADIUS, z0 = Math.sin(a) * RADIUS;
    const x1 = Math.cos(a + Math.PI) * RADIUS, z1 = Math.sin(a + Math.PI) * RADIUS;
    for (let i = 0; i < perRung; i++) {
      // Biased toward the ends: a real base pair reads as two bonded halves
      // with a gap, not an evenly filled bar.
      const raw = i / perRung;
      const u = raw < 0.5 ? raw * 0.8 : 1 - (1 - raw) * 0.8;
      write(
        x0 + (x1 - x0) * u + gauss() * 0.07,
        y + gauss() * 0.07,
        z0 + (z1 - z0) * u + gauss() * 0.07,
        0.5 + Math.random() * 0.35,
      );
    }
  }

  // ── dust ───────────────────────────────────────────────────────────
  for (let i = w; i < total; i++) {
    const a = Math.random() * Math.PI * 2;
    // Tight: the dust exists to soften the silhouette, not to fill the frame.
    // A wider spread throws particles across the copy column on the left, and
    // body text over moving specks is a legibility problem, not a mood.
    const r = RADIUS * (1.02 + Math.abs(gauss()) * 0.30);
    write(
      Math.cos(a) * r,
      (Math.random() - 0.5) * HEIGHT * 1.04,
      Math.sin(a) * r,
      0.3 + Math.random() * 0.35,
    );
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  /* aColor, NOT color.  Three injects its own `color` attribute declaration
   * into the vertex shader whenever vertexColors is on, and a ShaderMaterial
   * that also declares one fails to compile with a redeclaration error that
   * surfaces only as a blank canvas.  Renaming ours sidesteps it entirely. */
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aSize', new THREE.BufferAttribute(siz, 1));
  geo.computeBoundingSphere();
  return geo;
}

export const HELIX_HEIGHT = HEIGHT;
export const HELIX_RADIUS = RADIUS;

/* Repaint an existing cloud without rebuilding it.
 *
 * A theme flip changes only the colour ramp, and positions are the expensive
 * half of buildHelix — regenerating them would also reshuffle every particle,
 * so the helix would visibly jump on a change that is meant to be a recolour.
 */
export function recolour(geo, stops) {
  const pos = geo.getAttribute('position');
  const col = geo.getAttribute('aColor');
  const colourAt = gradient(stops);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    colourAt((pos.getY(i) + HEIGHT / 2) / HEIGHT, c);
    col.setXYZ(i, c.r, c.g, c.b);
  }
  col.needsUpdate = true;
}
