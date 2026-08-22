/* HarrisonGPT — helix hero controller.
 *
 * Owns the renderer, the scene and the single rAF loop.  Geometry and shaders
 * live in their own modules; scroll and step state arrive in the stages after
 * this one.
 *
 * Loaded as type="module", so it is deferred by default — the DOM is parsed
 * before this runs and there is no need to wait for DOMContentLoaded.
 */
import * as THREE from 'three';
import { buildHelix, recolour, HELIX_HEIGHT } from './geometry.js';
import { helixMaterial } from './material.js';
import { createScrollProgress } from './scroll-progress.js';
import { createSteps } from './steps.js';

const canvas = document.getElementById('helix-canvas');
const hero = document.querySelector('.helix-hero');
if (canvas && hero) boot();

/* Quality tier.  Chosen once at boot and never revisited: rebuilding 35k
 * particles mid-resize is a visible hitch, and a window that crosses a
 * breakpoint is far rarer than one that never does.
 *
 * The device-pixel-ratio cap matters more than the particle count.  Fill rate
 * is the bottleneck for a cloud of overlapping transparent sprites — every one
 * of them is drawn twice, core and halo — so uncapped DPR on a 3x phone screen
 * costs 9x the fragments for a difference nobody can see at that size.
 */
function pickTier() {
  const coarse = matchMedia('(pointer:coarse)').matches;
  // deviceMemory is Chromium-only and absent elsewhere; absent means "assume
  // fine", because defaulting to the low tier would penalise every Safari and
  // Firefox visitor for a missing API rather than for a slow machine.
  const lowMem = (navigator.deviceMemory ?? 8) <= 4;

  if (coarse || innerWidth < 700 || lowMem) return { name: 'low', quality: 0.35, dprCap: 1.5 };
  if (innerWidth < 1200) return { name: 'mid', quality: 0.6, dprCap: 1.75 };
  return { name: 'full', quality: 1, dprCap: 2 };
}

/* The helix palette and its blend mode are described in tokens.css, not here,
 * so a theme change is a stylesheet change.  Read once at boot; stage 7 makes
 * it react to a live theme switch. */
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    stops: [0, 1, 2, 3, 4].map((i) => v(`--helix-stop-${i}`)),
    additive: v('--helix-blend') === 'additive',
    coreAlpha: parseFloat(v('--helix-core-alpha')),
    haloAlpha: parseFloat(v('--helix-halo-alpha')),
  };
}

function boot() {
  const reduced = matchMedia('(prefers-reduced-motion:reduce)');

  /* The scrollytelling is set up first and unconditionally.  It is the part
   * that carries the actual content — headlines, cards, the rail — and it needs
   * no GPU.  Wiring it after the WebGL check meant a visitor without WebGL got
   * the poster *and* got stuck on step 01 with four steps they could not reach,
   * which is a worse failure than the one the poster exists to handle. */
  const progress = createScrollProgress(hero);
  const steps = createSteps(hero);

  /* Ask for the context ourselves rather than letting WebGLRenderer do it.
   *
   * Two reasons, both about the failure path.  three logs its own console
   * *error* before throwing, so catching the exception still leaves a red line
   * in the console of every visitor without WebGL — for a case this code
   * handles deliberately.  And the usual alternative, probing a throwaway
   * <canvas>, burns one of the browser's small pool of live GL contexts.
   * Passing the context in avoids both. */
  const attrs = { antialias: false, alpha: true, powerPreference: 'high-performance' };
  const gl = canvas.getContext('webgl2', attrs) || canvas.getContext('webgl', attrs);
  if (!gl) {
    // No WebGL, or a driver that refuses the context.  Show the CSS poster and
    // leave: a blank canvas would read as a broken page.
    document.querySelector('.helix-hero__poster')?.removeAttribute('hidden');
    canvas.hidden = true;
    hero.dataset.helix = 'poster';
    // Still drive the steps: without the loop nothing would ever call update().
    if (!reduced.matches) {
      const tick = () => { steps?.update(progress.raw); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }
    return;
  }
  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, ...attrs });
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  let theme = readTheme();
  const tier = pickTier();
  const dpr = Math.min(devicePixelRatio, tier.dprCap);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);

  // (spec) The camera pushes in and rises as the section is scrolled.  Small
  // numbers on purpose: the helix is meant to feel approached, not flown into.
  const CAM_Z0 = 20, CAM_Z1 = 16.5;
  const CAM_Y0 = 0, CAM_Y1 = 2.5;
  camera.position.set(0, CAM_Y0, CAM_Z0);

  const geo = buildHelix(theme.stops, tier.quality);

  // The bloom: one geometry, two draws.  The halo is the same points at ~3.4x
  // the size and a fraction of the alpha, so bright regions bleed into each
  // other exactly where particles are dense.
  // Sizes are pre-attenuation: the shader multiplies by 620/dist, which is
  // ~31 at the camera's resting distance.  0.062 therefore lands a core
  // particle at roughly 2 CSS px — small enough that 35k of them read as a
  // spray rather than a solid mass, which is what the first pass got wrong.
  const core = new THREE.Points(geo, helixMaterial({
    size: 0.062, alpha: theme.coreAlpha, additive: theme.additive, dpr,
  }));
  const halo = new THREE.Points(geo, helixMaterial({
    size: 0.21, alpha: theme.haloAlpha, additive: theme.additive, dpr,
  }));
  halo.renderOrder = 0;
  core.renderOrder = 1;

  const group = new THREE.Group();
  group.add(halo, core);
  // A slight lean stops the helix reading as a flat sine wave head-on.
  group.rotation.z = 0.14;
  scene.add(group);

  function resize() {
    // clientWidth/Height, not innerWidth: the canvas is inset:0 inside the
    // pinned viewport, which is not always the full window.
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);   // false: never write inline CSS size, the
    camera.aspect = w / h;           // stylesheet owns layout
    camera.updateProjectionMatrix();

    // Composition: the helix sits right of centre and leaves the left third
    // empty for the copy — but only when there is a left third to leave.  Below
    // the layout's 900px breakpoint the copy is over the canvas, so it centres.
    // Measured at CAM_Z0, not camera.position.z: the camera moves during
    // scroll, and deriving the offset from a live value would slide the helix
    // sideways as it pushes in.
    const visW = 2 * CAM_Z0 * Math.tan((camera.fov * Math.PI) / 360) * camera.aspect;
    group.position.x = w >= 900 ? visW * 0.13 : 0;
  }
  resize();
  addEventListener('resize', () => {
    resize();
    // No loop is running under reduced motion, so a resize would otherwise
    // leave a stretched frame from the old viewport on screen.
    if (reduced.matches) render(performance.now());
  }, { passive: true });

  /* Re-read the palette and the blend mode when the theme changes.
   *
   * This is the fragile path, so it is wired even though nothing in the UI
   * flips the theme today: additive blending only ever *adds* light, so the
   * exact material that glows on #08080A is invisible on paper.  Switching the
   * blend mode is not a nicety here, it is the difference between a hero and a
   * blank right-hand column — and a path that is never exercised is a path that
   * rots.  Set data-theme on <html> in devtools to try it.
   *
   * attributeFilter matters: without it this fires on every class change, and
   * the theme boot script removes .no-js from the same element at startup. */
  function applyTheme() {
    theme = readTheme();
    recolour(geo, theme.stops);
    const blending = theme.additive ? THREE.AdditiveBlending : THREE.NormalBlending;
    core.material.uniforms.uAlpha.value = theme.coreAlpha;
    halo.material.uniforms.uAlpha.value = theme.haloAlpha;
    // Blending is renderer state, not a shader define — no needsUpdate, which
    // would force a full program recompile for nothing.
    core.material.blending = blending;
    halo.material.blending = blending;
  }
  new MutationObserver(applyTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });

  function render(now) {
    const p = progress.step();
    // Undamped on purpose — see the note at the top of steps.js.
    steps?.update(progress.raw);

    // (spec) One and a half turns across the whole section, plus a slow
    // constant drift so the helix is alive before the reader scrolls at all.
    group.rotation.y = p * Math.PI * 2 * 1.5 + now * 0.00004;
    camera.position.z = CAM_Z0 + (CAM_Z1 - CAM_Z0) * p;
    camera.position.y = CAM_Y0 + (CAM_Y1 - CAM_Y0) * p;

    renderer.render(scene, camera);
  }

  let raf = 0;
  const loop = (now) => { raf = requestAnimationFrame(loop); render(now); };

  function start() {
    // Reduced motion gets exactly one frame: a still helix, no loop, nothing
    // that keeps a GPU awake for a reader who asked for none of it.
    if (raf || reduced.matches) return;
    // Snap the damping to where the page actually is.  Without this the helix
    // spins through however long the loop was asleep, all in one visible lurch.
    progress.settle();
    raf = requestAnimationFrame(loop);
  }
  function stop() {
    if (!raf) return;
    cancelAnimationFrame(raf);
    raf = 0;
  }

  /* Nothing about the loop is worth running while the section is off screen —
   * and on this page the reader spends most of their scroll below it. */
  new IntersectionObserver(
    ([e]) => (e.isIntersecting ? start() : stop()),
  ).observe(hero);

  // Background tabs throttle rAF but do not reliably stop it, and a tab that
  // has been hidden for a minute should not resume mid-spin.
  addEventListener('visibilitychange', () => {
    if (document.hidden) stop(); else if (isOnScreen()) start();
  });

  function isOnScreen() {
    const r = hero.getBoundingClientRect();
    return r.bottom > 0 && r.top < innerHeight;
  }

  // Honour a live change to the setting, both directions.
  reduced.addEventListener('change', () => {
    stop();
    progress.measure();
    resize();
    if (reduced.matches) render(performance.now()); else if (isOnScreen()) start();
  });

  if (reduced.matches) render(performance.now()); else start();

  /* Read-only handle for verification — probe scripts and devtools.  Getters
   * only: nothing is written to it per frame, so it costs nothing to leave in.
   * It is the only way to assert that scroll actually drives the scene without
   * diffing pixels, and a rotation that silently stops updating is precisely
   * the kind of regression a screenshot will not catch. */
  window.__helix = {
    get raw() { return progress.raw; },
    get damped() { return progress.damped; },
    get rotY() { return group.rotation.y; },
    get camZ() { return camera.position.z; },
    get camY() { return camera.position.y; },
    get blending() { return core.material.blending; },
    get alphas() { return [core.material.uniforms.uAlpha.value, halo.material.uniforms.uAlpha.value]; },
    get colour0() { const a = geo.getAttribute('aColor'); return [a.getX(0), a.getY(0), a.getZ(0)]; },
    get tier() { return tier.name; },
    get running() { return raf !== 0; },
  };

  // A one-word answer to "did the module graph resolve?" that does not depend
  // on reading pixels.
  hero.dataset.helix = 'ready';
  hero.dataset.helixCount = String(geo.getAttribute('position').count);
  hero.dataset.helixTier = tier.name;
}

export { HELIX_HEIGHT };
