# Interface audit · September 2026

The lab had accumulated four competing control bands above the TinyStories canvas, two fixed narrow sidebars, and labels as small as 7–9 px. Initialization, live generation, batch sampling, projection diagnostics, and experiment history all appeared at once. Researchers had to read the instrument's implementation before finding the model.

## First-pass hierarchy

- **Navigation:** one lab-wide navigation row. Subword/character selection sits beside the TinyStories title; it no longer occupies a separate navigation band.
- **Model context:** parameter count, layer count, and the measured checkpoint remain visible above the workspace. The canvas toolbar controls only the visualization.
- **Workspace:** one large canvas and one adjustable tools panel. Model & runs contains saved specimens, initialization, compute options, and training. Inspect contains prompt probing, exact channel addresses, measured values, and interventions. Selecting a point opens its inspector.
- **Progressive disclosure:** initialization, training configuration, corpus provenance, projection audit, map history, batch sampling, and detailed learning evidence have named disclosures. Live output stays beside the network. A concise projection interpretation remains visible even when the numerical audit is closed.
- **Reading:** controls and metadata have a 12 px minimum, with larger prose and headings. Color remains desaturated, surfaces use thin rounded borders, and panels have no side shadows.
- **History:** the field journal has a reading column with expandable entries and a separate archive column. Query studies disclose their protocol while keeping results prominent. Methods use readable two-column articles and identify the study they describe.

## Panel interaction

Drag the divider beside the tools panel to resize it. Keyboard users can focus the resize slider and use Left/Right in 24 px increments, or Home/End for its bounds. Double-click restores the default 360 px width. Width is bounded by the available canvas space; collapsed state and preferred width are stored locally per workspace. Storage failures do not block the lab.

Collapsing tools gives the canvas the available width. Switching Model/Inspect preserves the resident model and measurements. Below 1,000 px, tools become a collapsible section above the canvas; the pointer-only divider disappears. Optional content remains accessible through the same buttons and disclosures.

## Corrections after use

The first pass placed evidence after the entire workspace grid. A tall tools panel therefore left a blank area beneath a shorter visualization. Each workspace now keeps its evidence in the canvas column, immediately after the visualization, regardless of panel height.

Generation and intervention prerequisites were also too obscure. TinyStories now offers **Load & generate live**, which restores the current specimen's weights or loads the explicitly identified trained example before generating. The displayed prefix is preserved. The button then becomes Pause/Resume as before. Recorded replay remains a separate operation.

The binding model's **Intervention** control opens a short explanation and a **Measure 256 interventions** action when its causal measurements are missing. TinyStories' unit inspector explains whether it needs weights, a current prompt measurement, a selected unit, or stopped playback, and offers the relevant preparation action. These paths use the existing measured forward passes and archive contracts.

The correction passed seven browser workflows, including a fresh one-action model load/generation followed by a measured single-unit ablation. A layout assertion checks a gap of at most 20 px in both tools-panel views, and the new intervention-map action computes real effects. Type checking and the static Pages worker/checkpoint smoke test also passed.

## First-pass verification scope

The UI regression suite exercises panel resizing with both pointer and keyboard, collapse/expand, persisted width, measured replay values across view changes, light mode, and a narrow viewport with evidence expanded. Existing end-to-end tests follow the new disclosure paths for training, prompt probes, intervention, export/import, and resume. The Pages smoke test also exercises those paths against the actual static deployment.

Validation for this revision:

- 170 numerical and archive tests passed across 20 files.
- All 14 browser workflows passed, including continuous training beyond 125 updates and checkpoint export at the actual paused step.
- Type checking reported no errors or warnings; formatting, lint, and the Svelte component audit passed.
- The production Pages build passed its strict static-server smoke test: BPE replay, BPE and character checkpoint restoration with new WASM prompt probes, and binding/paired-query reference loading. There were no page errors or failed requests.
- The workspace test retained unit 1403, replay frame 11, and the measured activation 1.8640 through panel changes. Desktop dark/light views and a 390 px viewport were reviewed; expanded evidence produced no horizontal page overflow.
- The refreshed 3,200 × 2,240 README banner is an actual browser capture of the published step-4096 model. Its measurement source and image hash are recorded in `docs/assets/tissue-banner.json`.

This is an instrument design change, not a new scientific result. Training code, recorded activations, projection algorithms, checkpoint weights, and historical provenance are unchanged. The 3D map remains PCA; similarity edges do not represent model connections. Paced layer playback still presents measured values rather than GPU execution timing.

## Action-first revision · September 18

The first two passes improved spacing and prerequisite messages but left the main experiments near the bottom. That failed the central task: running a model while watching its network. The revised hierarchy starts with actions:

- **TinyStories:** Generate / Pause / Resume, Replay trace, Train / Pause training, Probe, Intervene and Model share one study action bar. A fresh visit offers Replay example, which loads and plays the measured reference without allocating weights. Once a different specimen is open, replay never silently replaces it.
- **Console:** the prompt, emitted text, layer transport and selected raw value sit above the canvas. Playback settings, recorded-token selection and full probability evidence expand in place. Sampling parameters and batch comparison live under Model → Generation settings.
- **Tools:** the TinyStories tools panel starts closed. Model, Probe and Intervene open the appropriate content; the panel header is a title and close control rather than a second navigation bar. Resizing remains available. Intervene puts preparation and ablation controls directly below the unit picker, preserving the reason an action may be unavailable.
- **Binding:** training, probing, intervention-map setup and repair setup are available in its action bar. Repair explains whether an intervention atlas or a selected lesion target is needed before comparison.
- **Continuity:** Generate keeps one button element while its action changes to Pause or Resume. Opening tools preserves the selected unit and measured replay frame. Projection caveats remain beside the network, and evidence still follows the canvas without inheriting the sidebar height.

The character-model comparison retains its existing study controls. This revision concentrates the new interaction hierarchy on the live subword lab and binding workbench; it does not change model computation or historical measurements.

A laptop regression checks that the action bar, full network and layer transport are in view at 1,280 × 850, and that Intervene reveals its controls without scrolling through the inspector. Desktop/light/mobile browser captures accompany the replay, resizing, generation, training and ablation workflows. The README banner is recaptured from the actual published trained specimen.

Validation for the action-bar revision: all 17 browser workflows passed across the initial run and the corrected rerun; the six workflows covering final layout and navigation changes were rerun together. These include actual checkpoint loading, live generation, pause/step/stop, exact single-unit ablation, continuous training, archive round-trips, and a retained replay value of 1.8640 for unit 1403. Svelte/type checks reported no errors or warnings, the component autofixer returned no issues or suggestions, and lint passed. The GitHub Pages build passed its strict static-server replay, WASM checkpoint restoration and worker-path checks with no page errors or failed requests.
