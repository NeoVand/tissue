# First milestone validation

Recorded on 2026-09-16. These checks establish a working measurement instrument; they do not establish the usefulness of its geometry for interpretability.

| Check                                  | Result                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| TypeScript and Svelte diagnostics      | No errors or warnings                                        |
| Svelte component autofixer             | No remaining issues or suggestions                           |
| Formatting and ESLint                  | Passed                                                       |
| Numerical, geometry, and journal tests | 66 passed                                                    |
| Production browser tests               | 3 passed                                                     |
| Production app and worker build        | Passed; renderer bundle exceeds the default size advisory    |
| JaxJS runtime doctor                   | 15 contracts passed                                          |
| Desktop and mobile visual inspection   | No horizontal overflow; actual measured coordinates rendered |

The browser tests exercise reference loading, token and neuron inspection, a real four-arm repair, historical checkpoint inspection, notes, JSON export, reload and resume, training and pause, and mobile navigation. A third test injects a capture failure after 50 real training updates, confirms numerical actions are blocked, exports the interrupted evidence, and verifies recovery into a separate run from the last durable checkpoint.

The unit tests include actual ablation, checkpoint continuation, frozen repair weights, PCA and alignment, original-space neighbors, undefined fingerprints, and invalid or corrupt record rejection. After excluding undefined directional fingerprints, the recorded neuron-172 repair retains exactly the same neighborhood membership for every directional selector.

Two independent WebGPU training runs reached 97.92% (seed 42) and 100% (seed 7) on the fixed 96-prompt evaluation set at 2,000 updates. Raw captures and provenance are preserved in [the experiment archive](../static/experiments/README.md). The field journal explains the long learning plateau, the synchronous GPU readback failure found during development, projection distortion, and the inconclusive repair pilot.

The first repair pilot uses a weak lesion, one seed, and unmatched random controls. It lacks an unlesioned fine-tuning control and does not demonstrate a superior functional geometry. Those limitations and the next experiment are visible in the application.
