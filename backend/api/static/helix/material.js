/* Shaders for the helix.
 *
 * Two things here are fake on purpose, and both are the reason this hero does
 * not need `examples/jsm` vendored alongside three:
 *
 *   bloom — the same geometry is drawn twice.  Once as the core, once
 *           oversized and nearly transparent as a halo.  A real
 *           UnrealBloomPass costs two extra render targets and a blur chain
 *           for an effect that, on a static-coloured point cloud, is
 *           indistinguishable from this.
 *   depth of field — computed per vertex and applied as a widened falloff in
 *           the fragment shader.  A real DOF pass needs a depth texture.
 *
 * Blending is a runtime input, not a constant.  Additive blending cannot
 * darken: it only ever adds light, so a helix that glows on #08080A washes out
 * to nothing on a paper ground.  The light theme therefore renders with
 * NormalBlending and roughly a third of the halo alpha.  Both come from CSS
 * custom properties so the two themes stay described in one place — tokens.css.
 */
import * as THREE from 'three';

const VERT = /* glsl */ `
  uniform float uSize;
  uniform float uDpr;
  uniform float uFocus;
  uniform float uFocusRange;

  attribute vec3 aColor;
  attribute float aSize;

  varying vec3 vColor;
  varying float vBlur;

  void main() {
    vColor = aColor;

    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float dist = -mv.z;

    // (spec) 620.0 / dist is the verified attenuation constant.  Get this
    // wrong and gl_PointSize lands below one pixel: the draw call succeeds,
    // the GPU reports nothing, and the canvas is simply empty.  uDpr is
    // separate because gl_PointSize is in device pixels while uSize is
    // authored in CSS pixels.
    gl_PointSize = uSize * aSize * uDpr * (620.0 / dist);

    vBlur = clamp(abs(dist - uFocus) / uFocusRange, 0.0, 1.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAG = /* glsl */ `
  uniform float uAlpha;

  varying vec3 vColor;
  varying float vBlur;

  void main() {
    // gl_PointCoord is 0..1 across the sprite; square points would read as a
    // grid of pixels the moment they grow past a few px.
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;

    // Out-of-focus particles get a wider, softer falloff and less opacity.
    float edge = mix(0.40, 0.06, vBlur);
    float a = smoothstep(0.5, 0.5 - edge - 0.02, d);

    gl_FragColor = vec4(vColor, a * uAlpha * mix(1.0, 0.34, vBlur));

    // ShaderMaterial opts out of three's automatic output conversion, so the
    // linear working-space colour has to be encoded here or the whole helix
    // renders visibly darker and more saturated than the tokens describe.
    #include <colorspace_fragment>
  }
`;

/**
 * @param {object} o
 * @param {number} o.size        base point size in CSS px at the focal plane
 * @param {number} o.alpha       overall opacity
 * @param {boolean} o.additive   true in dark themes only
 * @param {number} o.dpr
 */
export function helixMaterial({ size, alpha, additive, dpr }) {
  return new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uSize: { value: size },
      uAlpha: { value: alpha },
      uDpr: { value: dpr },
      // Focus sits a little in front of the helix axis so the near face is
      // sharp and the far one falls away; range is roughly the cloud's depth.
      uFocus: { value: 18.0 },
      uFocusRange: { value: 11.0 },
    },
    transparent: true,
    // No depth buffer participation at all.  These are additive-ish glowing
    // sprites with no interior: writing depth would make near particles punch
    // opaque holes in the ones behind them.
    depthWrite: false,
    depthTest: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}
