/** Analytic sphere impostors and measured-coordinate interpolation, with no procedural motion. */
export const nodeVertex = /* glsl */ `
attribute vec3 fromPosition;
attribute vec3 toPosition;
attribute vec4 radiusAlpha;
attribute vec3 fromColor;
attribute vec3 toColor;
attribute vec2 emphasis;
attribute float layerIndex;
attribute float activityLevel;
uniform float progress;
uniform float viewportHeight;
uniform float activeLayer;
uniform float activityMode;
uniform float brightnessMode;
varying vec2 disc;
varying vec3 beadColor;
varying float alpha;
varying float selected;
varying float signal;
varying float layerFocus;
void main() {
  vec3 center = mix(fromPosition, toPosition, progress);
  vec4 viewCenter = modelViewMatrix * vec4(center, 1.0);
  float radius = mix(radiusAlpha.x, radiusAlpha.y, progress);
  float focused = activeLayer < -0.5 || abs(layerIndex - activeLayer) < 0.5 ? 1.0 : 0.0;
  float minimumPixels = mix(mix(4.4, 2.4, activityMode), 3.4, brightnessMode);
  float minimumRadius = minimumPixels * max(0.001, -viewCenter.z) / (viewportHeight * projectionMatrix[1][1]);
  radius = max(radius, minimumRadius);
  radius *= mix(mix(0.62, 1.0, focused), 1.0, brightnessMode);
  disc = position.xy * mix(1.65, 2.2, brightnessMode);
  viewCenter.xy += disc * radius;
  gl_Position = projectionMatrix * viewCenter;
  beadColor = mix(fromColor, toColor, progress);
  alpha = mix(radiusAlpha.z, radiusAlpha.w, progress) * mix(mix(0.075, 1.0, focused), mix(0.65, 1.0, focused), brightnessMode);
  selected = mix(emphasis.x, emphasis.y, progress);
  signal = activityLevel;
  layerFocus = focused;
}
`;

export const nodeFragment = /* glsl */ `
uniform float dark;
uniform float brightnessMode;
varying vec2 disc;
varying vec3 beadColor;
varying float alpha;
varying float selected;
varying float signal;
varying float layerFocus;
void main() {
  float radius = length(disc);
  // One screen pixel of feathering keeps small cores crisp at normal display density.
  float aa = max(0.5 * fwidth(radius), 0.008);
  float body = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, radius);
  float ring = (1.0 - smoothstep(0.035, 0.035 + aa, abs(radius - 1.42))) * selected;
  float coverage = max(body, ring * 0.75) * alpha;
  if (brightnessMode > 0.5) {
    // The measured value changes intensity, never core or halo-envelope dimensions.
    // A nonzero structural floor keeps zero-activation and nonfocused units visible.
    float energy = clamp(signal, 0.0, 1.0) * mix(0.18, 1.0, layerFocus);
    float halo = (1.0 - smoothstep(1.0, 2.2, radius)) * smoothstep(0.65, 1.0, radius);
    float coreAlpha = body * (0.44 + 0.56 * energy) * alpha;
    float glowAlpha = halo * energy * mix(0.16, 0.28, dark) * alpha;
    coverage = max(max(coreAlpha, glowAlpha), ring * 0.80 * alpha);
    if (coverage < 0.008) discard;
    vec3 coreColor = mix(beadColor * 0.55, beadColor * 0.90 + vec3(0.28 * dark), energy);
    vec3 glowColor = mix(beadColor * 0.68, beadColor * 0.65 + vec3(0.35), dark);
    gl_FragColor = vec4(mix(coreColor, glowColor, smoothstep(0.95, 1.08, radius)), coverage);
    #include <colorspace_fragment>
    return;
  }
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
attribute vec2 endpointLayers;
uniform float progress;
uniform float activeLayer;
varying float edgeAlpha;
varying float selected;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(mix(fromPosition, toPosition, progress), 1.0);
  float focused = activeLayer < -0.5 || (abs(endpointLayers.x - activeLayer) < 0.5 && abs(endpointLayers.y - activeLayer) < 0.5) ? 1.0 : 0.0;
  edgeAlpha = mix(opacity.x, opacity.y, progress) * mix(0.08, 1.0, focused);
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
