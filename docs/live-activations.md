# Watching a forward pass

TinyStories → Subword now couples autoregressive generation to measured MLP
activations. It is a viewing instrument, not evidence of a new circuit or an
analogy to biological timing.

## What is measured

For each generated token, the worker performs a causal forward pass on the exact
current token-ID context. It reads every post-ReLU MLP channel at the final real
input position and the raw next-token probability distribution from that same
forward pass. The frame records the model identity, checkpoint step, training
counter, backend, tokenizer, exact context IDs and pieces, input position, raw
activations, probabilities, and sampled output token.

These activations belong to the **input position that predicts the next token**.
The new output token enters the context on the following forward pass. For
example, the activity while processing `little` may produce ` girl`; the activity
for ` girl` is measured on the next pass. Generated token IDs are carried forward
directly. Decoding and re-encoding the growing string could merge pieces and
would change the model's actual input, so the live path never does that.

The view shows the MLP channels already represented in the model atlas. It does
not separately visualize attention heads, residual-stream coordinates, or every
activation inside a transformer block. Each displayed unit keeps its exact
layer/channel and incoming/outgoing tensor addresses. Zero is a measured value;
faint structural points do not imply activity.

## Live playback and ordering

The worker sends one provisional frame and waits for the interface to acknowledge
it. The interface presents the measured MLP values in model order, L1 through the
last layer, then reveals the sampled output token. Only after acknowledgment may
the worker add that token to its context and compute the next frame. The viewing
pace therefore bounds generation instead of playing an animation over a sentence
that was already generated in full.

The layer sequence is **paced playback of one measured forward pass**. Its dwell
time is chosen for inspection, not measured kernel duration. The GPU may execute
and fuse operations much faster. Units within a layer are not given invented
individual firing times, and interlayer attention/residual computation is not
replaced by spatial edges.

Pause holds the current frame. Layer and token stepping advance the playback at
explicit boundaries. Stop cancels the current provisional frame and retains only
acknowledged output tokens with their matching frames. Stopping before the first
acknowledgment produces an empty, cancelled completion. Computation has a watchdog;
waiting for the user to advance a paused frame does not time out. Teardown cancels
pending acknowledgments and terminates the worker after a bounded grace period.

The architectural layout makes layer order easy to follow and includes every MLP
channel. Functional positions remain the checkpoint's fixed calibration map;
they are not fitted anew to each generated token. That map still uses three PCA
coordinates from 128-dimensional activation fingerprints. Token progression is
the temporal axis of playback, not a fourth PCA component. UMAP and ICA are not
implemented by this change. The PCA solver retains a fourth eigenvalue to check
separation at the three-component boundary; it is not a displayed coordinate.

The renderer focuses the current layer with shader uniforms, so advancing layers
does not restart spatial interpolation or upload unchanged neuron geometry.
During activity playback, point size and opacity use the square root of activation
relative to the largest visible activation in that frame. This helps expose sparse
responses; brightness is not an absolute scale across tokens. The inspector and
saved trace retain unscaled values. Nonfocused layers are dimmed for orientation,
and similarity edges remain similarity edges rather than paths of signal flow.
Live activation values switch directly between measured frames. They are not
interpolated across tokens; this prevents fast playback from displaying blended
activity as if it belonged to the current input. Switching between functional
and architectural layouts fits the new coordinates once; advancing tokens or
layers preserves the camera.

## Records and compatibility

A completed or stopped generation retains every acknowledged frame, up to the
chosen 256-token generation limit. Its `liveTraces` entry refers to the sample's
index in the same run. Existing binary export/import and IndexedDB storage retain
the typed activation and probability arrays, with additional validation for exact
context transitions, dimensions, numerical values, token labels, checkpoint
identity, EOS and cancellation. Older samples without traces remain readable;
the interface does not invent their missing measurements.

The Study 004 numerical implementation remains unchanged. New `live-*` modules
add the streaming path, so source hashes and independent audits of the earlier
measured reference remain valid. Live and batch generation use the same sampling
rules, but a different compiled capture path can introduce floating-point
variation. Sampling seed alone is not a promise of bitwise identical text across
backends. Parameters, optimizer state and training randomness must remain
unchanged while generating or replaying.

## Continuous training

Continuous mode repeats short worker bursts until Pause. The worker can receive a
stop request between optimizer updates. Metrics arrive between captures; a map
and complete checkpoint are saved every 100 updates and when training pauses.
Finite 25/100/500-update bursts remain available. A pause during measurement waits
for a consistent capture rather than treating half-captured evidence as complete.

Continuous means there is no chosen total update budget. Browser resources and
storage remain finite. Save failures are surfaced, and recovery uses the last
complete checkpoint. Training and live generation are serialized on the resident
model; a generation trace always refers to one fixed checkpoint.

## Recorded example

Open **Live replay · TinyStories BPE small at 4096** under Recorded specimens,
then choose **Replay trace** or a recorded token. This 5.90 MB inspection-only
archive requires no model allocation. Its 24 WebGPU frames were captured from the
original 1.85M-parameter checkpoint with prefix `Once upon a time`, seed 71,
temperature 0.8 and top-k 40. The completion is:

> , there was a little girl called Tom. He had a great idea and a flower. One day, he was feeling

Every frame contains 2,048 measured channel values and the full 4,096-token
distribution. The original four calibration maps, training metrics, five earlier
samples and intervention are inherited unchanged and explicitly identified as
such. This is an inference capture, not a new trained replication. The sample's
character inconsistency remains visible. Its longest exact training-story span
is seven tokens; that descriptive overlap is not a coherence or originality test.

The [capture report](../static/experiments/token-stories-live-demo-d474f204-report.json)
records source hashes and exact checkpoint preservation. The
[archive validation receipt](../static/experiments/token-stories-live-demo-validation.json)
checks the published trace, probability mass and inherited evidence. The original
resumable reference remains available; the complete replay archive with weights
is also retained locally at the path recorded in the report.

```sh
# With pnpm dev running; restores published weights and captures 24 token frames.
node scripts/measure-live-stories.mjs --backend webgpu --publish
node scripts/audit-token-stories.mjs --write
node scripts/analyze-token-story-samples.mjs
```

## Verification

The change passed 165 unit tests and the full 12-flow production browser suite.
After the final compact-layout, automatic-framing and transparent-depth fixes,
the three affected live/subword browser flows passed again. TypeScript/Svelte
checks, formatting, lint and component autofixers were clean.

The implementation is checked against raw forward-pass values, exact sliding
contexts, controlled EOS generation, cancellation and checkpoint preservation.
Browser checks exercise real worker backpressure, layer/token controls, stopped
sample persistence and replay, continuous training beyond a finite burst, and
restoration. The production continuous-training check passes 125 updates, then
compares the paused metric, captured map and exported checkpoint steps.

A separate full Small-model WASM check found zero difference between the first
live frame and a direct prompt probe's activation/probability values. Four live
sample IDs matched batch generation in that run. Active and queued cancellation,
backpressure, typed archive round-trip and exact preservation of weights, Adam
state and training RNG passed. The
[engineering receipt](../static/experiments/token-stories-live-engine-smoke.json)
records the backend, source hashes and local complete artifact. This engineering
check is not a language-quality experiment.

The [production screenshot](assets/live-token-activations.png) shows a paused
first-token forward pass, after advancing to L2. The selected U1024 channel belongs
to L3 and reads 0.44004 in that measured pass; layer focus does not change its raw
value. The completed sample below the graph is explicitly a separate saved batch
sample. Desktop dark/light and 390-pixel mobile layouts were checked, including
automatic framing when switching from functional coordinates to model layout.

A real WebGL framebuffer check verifies that faint foreground context cannot
hide focused activity behind it, and that leaving activity mode restores normal
snapshot rendering. The [renderer receipt](../static/experiments/live-field-renderer-check.json)
retains measured pixels and source hashes. Re-run it with the development server
open using `node scripts/verify-live-field.mjs`.
