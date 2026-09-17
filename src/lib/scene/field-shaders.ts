/** Analytic sphere impostors and measured-coordinate interpolation, with no procedural motion. */
export const nodeVertex = /* glsl */ `
attribute vec3 fromPosition;
attribute vec3 toPosition;
attribute vec4 radiusAlpha;
attribute vec3 fromColor;
attribute vec3 toColor;
attribute vec2 emphasis;
uniform float progress;
uniform float viewportHeight;
varying vec2 disc;
varying vec3 beadColor;
varying float alpha;
varying float selected;
void main() {
  vec3 center = mix(fromPosition, toPosition, progress);
  vec4 viewCenter = modelViewMatrix * vec4(center, 1.0);
  float radius = mix(radiusAlpha.x, radiusAlpha.y, progress);
  float minimumRadius = 4.4 * max(0.001, -viewCenter.z) / (viewportHeight * projectionMatrix[1][1]);
  radius = max(radius, minimumRadius);
  disc = position.xy * 1.65;
  viewCenter.xy += disc * radius;
  gl_Position = projectionMatrix * viewCenter;
  beadColor = mix(fromColor, toColor, progress);
  alpha = mix(radiusAlpha.z, radiusAlpha.w, progress);
  selected = mix(emphasis.x, emphasis.y, progress);
}
`;

export const nodeFragment = /* glsl */ `
uniform float dark;
varying vec2 disc;
varying vec3 beadColor;
varying float alpha;
varying float selected;
void main() {
  float radius = length(disc);
  // One screen pixel of feathering keeps small cores crisp at normal display density.
  float aa = max(0.5 * fwidth(radius), 0.008);
  float body = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radius);
  float ring = (1.0 - smoothstep(0.035, 0.035 + aa, abs(radius - 1.42))) * selected;
  float coverage = max(body, ring * 0.75) * alpha;
  if (coverage < 0.008) discard;
  vec3 normal = vec3(disc, sqrt(max(0.0, 1.0 - dot(disc, disc))));
  float diffuse = max(0.0, dot(normal, normalize(vec3(-0.45, 0.65, 1.0))));
  float specular = pow(max(0.0, dot(normal, normalize(vec3(-0.32, 0.42, 1.6)))), 28.0);
  float rim = pow(1.0 - normal.z, 3.0);
  vec3 shaded = beadColor * (0.50 + diffuse * 0.50 + dark * rim * 0.16) + vec3(specular * 0.20);
  gl_FragColor = vec4(mix(shaded, beadColor, step(1.05, radius)), coverage);
  #include <colorspace_fragment>
}
`;

export const edgeVertex = /* glsl */ `
attribute vec3 fromPosition;
attribute vec3 toPosition;
attribute vec2 opacity;
attribute vec2 emphasis;
uniform float progress;
varying float edgeAlpha;
varying float selected;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(fromPosition, toPosition, progress), 1.0);
  edgeAlpha = mix(opacity.x, opacity.y, progress);
  selected = mix(emphasis.x, emphasis.y, progress);
}
`;

export const edgeFragment = /* glsl */ `
uniform vec3 baseColor;
uniform vec3 selectedColor;
varying float edgeAlpha;
varying float selected;
void main() {
  gl_FragColor = vec4(mix(baseColor, selectedColor, selected), edgeAlpha);
  #include <colorspace_fragment>
}
`;
