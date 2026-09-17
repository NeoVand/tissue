# GitHub Pages deployment

The public lab is hosted at **https://neovand.github.io/tissue/**. GitHub Pages
serves the static application and its published data; model training, inference,
geometry analysis, and personal experiment storage run in the visitor's browser.

## Build and publish

The repository's Pages source is **GitHub Actions**. The
[Pages workflow](../.github/workflows/pages.yml) runs on pushes to `main` and can
also be dispatched manually. Its build job installs the pinned pnpm version on
Node 24, installs from the lockfile, checks the project and numerical/archive
contracts, builds the static site, and tests its browser flows before uploading
the `build/` artifact. A separate deployment job publishes it to the `github-pages`
environment with `pages: write` and `id-token: write`. Action versions are pinned
to commit hashes.

```sh
pnpm install --frozen-lockfile
pnpm check
pnpm build:pages
pnpm preview:pages
# Open http://localhost:4180/tissue/
```

`TISSUE_DEPLOY_TARGET=github-pages` selects SvelteKit's static adapter and the
`/tissue` base path. The root page is prerendered; a `404.html` fallback is included.
`pnpm dev` continues to use `/`, while `pnpm build` and `pnpm preview` retain the
original Cloudflare target. There are no inference-server credentials to deploy.

Dataset and specimen paths stored in research records remain canonical, such as
`/experiments/token-stories-….tissue`. The shared `publicAsset` helper adds the
hosting base **only when requesting a file**, in both the interface and model
workers. Reference validation still checks canonical filenames, part ordering,
declared byte counts, SHA-256, and model/tokenizer identity. Corpus bytes,
tokenizers, checkpoints, and recorded measurements are unchanged.

## Verify the static site

```sh
pnpm exec playwright install chromium
node scripts/verify-pages.mjs
# After deployment, repeat against the public site:
node scripts/verify-pages.mjs --url https://neovand.github.io/tissue/
```

The local check serves the actual `build/` files beneath `/tissue/`, with strict
404s and no server rendering. It opens the 24-frame BPE replay, restores the
trained BPE and character checkpoints using WASM, computes new prompt probes,
and opens the controlled-task and paired-query references. It checks request
paths, failed downloads, and browser errors. It performs no training. The
workflow installs Chromium and its Linux dependencies before this check.

## Historical source audits

Published experiments record exact source hashes. This deployment changes the
fetch URLs in the two dataset loaders and two archive loaders, and adds a static
adapter to the lockfile. Their hashes consequently differ from the originals;
deployment is not a new training result. Historical receipts are preserved as
recorded rather than rewritten to match new code.

Commit **`73cf9d9e3f4e9f1fc4149e8321ad17e10a8e74ea`** contains the predeployment
sources. Its bytes match all 22 source hashes of the original BPE reference,
17 hashes of the compact character reference, and 28 hashes of the live replay
capture. To run a strict historical source audit without changing your working
checkout:

```sh
git worktree add --detach ../tissue-study004 73cf9d9e3f4e9f1fc4149e8321ad17e10a8e74ea
cd ../tissue-study004
pnpm install --frozen-lockfile
node scripts/audit-token-stories.mjs
node scripts/audit-stories.mjs
```

Run historical capture scripts there as well when they require the parent's
exact source hashes. Fresh runs from the deployed lab retain their own raw data
and local history; viewing an old specimen does not retrain it or change its
provenance.

## README image

[The banner](assets/tissue-banner.png) is a real **3,200 × 2,240** browser screenshot
of the published 1.85M-parameter BPE model at step 4,096. It shows a paused measured
replay on the functional PCA map. It is not an illustration or composited UI.
[Capture metadata](assets/tissue-banner.json) records the specimen checksum,
selected frame and layer, image dimensions, and image hash.

Recreate it against a running local build:

```sh
pnpm exec playwright install chromium
node scripts/capture-readme-banner.mjs --url http://localhost:4180/tissue/
```

The script loads the published specimen through the interface, pauses its replay,
selects a measured channel, adjusts the camera using its existing controls, and
captures the viewport at device pixel ratio 2. No model data or page markup is
injected. Capture the actual visible canvas: offscreen rendering is intentionally
suspended to save GPU work.

Deployment follows the official
[SvelteKit static-hosting guidance](https://svelte.dev/docs/kit/adapter-static#GitHub-Pages)
and [GitHub Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
