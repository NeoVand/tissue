<script lang="ts">
	import Icon from './Icon.svelte';
</script>

<div class="methods">
	<header>
		<span class="eyebrow">Instrument specification / v0.2</span>
		<h1>Methods & limitations</h1>
		<p>What each view measures, what it leaves out, and what would count as progress.</p>
	</header>
	<div class="grid">
		<article>
			<Icon name="network" />
			<h2>Model and task</h2>
			<p>
				Two causal transformer blocks, width 32, four attention heads, and 128 ReLU MLP units per
				block. There are 25,920 parameters. The model predicts an answer token after a 14-token
				variable-binding prompt.
			</p>
			<code>a=3;b=7;c=2;?b → 7</code>
			<p>
				Whole assignments are disjoint across training, calibration, and evaluation. The fixed
				held-out set contains 96 prompts. Uniform value guessing scores 12.5%; randomly copying one
				supplied value scores 33.3%.
			</p>
		</article>
		<article>
			<Icon name="cube" />
			<h2>Every coordinate has a source.</h2>
			<p>
				In functional space, a point is one post-ReLU MLP channel. Activation fingerprints
				concatenate responses across 16 calibration prompts and all 14 positions. We center and
				normalize each fingerprint and project it with PCA.
			</p>
			<p>
				The model-layout view places those same channels by layer and index. Its coordinates are
				architectural labels, not measured functional distances. Attention heads and residual
				dimensions are shown in the architecture diagram, not represented as MLP points.
			</p>
		</article>
		<article>
			<Icon name="target" />
			<h2>Intervention and direction</h2>
			<p>
				Each intervention fingerprint is the change in eight answer probabilities after silencing
				one neuron at every token position, across calibration prompts. Effects are normalized for
				direction; raw magnitude is retained separately.
			</p>
			<p>
				Zero-variance activation and zero-effect fingerprints have no defined direction. They are
				omitted from functional maps and excluded from directional repair neighborhoods. Similarity
				edges are not computational connections.
			</p>
		</article>
		<article>
			<Icon name="activity" />
			<h2>Projection and motion</h2>
			<p>
				Neighbor retention reports how many original-space neighbors survive projection. Explained
				variance reports the fraction captured by three components. Neither metric establishes
				interpretability.
			</p>
			<p>
				Successive maps are aligned by an orthogonal transformation. Shader interpolation makes
				recorded transitions readable; intermediate display frames are not measurements.
				Reduced-motion preferences disable interpolation.
			</p>
		</article>
		<article>
			<Icon name="flask" />
			<h2>Controlled repair pilot</h2>
			<p>
				Choose eight same-layer units using activation, intervention, outgoing-weight, or random
				neighborhoods. Freeze all other weights; clamp the lesion throughout 50 training updates.
				Each arm starts from identical weights, optimizer initialization, and batches.
			</p>
			<p>
				The first lesion caused little damage. The pilot lacks strength-matched random units and
				unlesioned fine-tuning controls, and does not demonstrate a superior geometry. The source
				model remains unchanged.
			</p>
		</article>
		<article>
			<Icon name="book" />
			<h2>Reproducibility and scope</h2>
			<p>
				JSON exports preserve seeds, architecture version, backend, metrics, latest raw
				fingerprints, checkpoint weights, optimizer state, training RNG, recorded maps, probes,
				interventions, and notes. Earlier maps do not contain earlier weights.
			</p>
			<p>
				When a capture fails after training, later observed metrics remain in the interrupted
				record. Resuming durable weights creates a separate recovered run. GPU arithmetic can vary
				across devices.
			</p>
		</article>
	</div>
	<div class="next">
		<span class="eyebrow">Research direction</span>
		<h2>Can the map predict an unseen intervention?</h2>
		<p>
			We have two reproducible learning transitions and an inconclusive repair pilot. The next
			useful result would be a preregistered prediction: use calibration fingerprints to choose a
			neighborhood, then test recovery or intervention effects on held-out assignments against
			matched controls. An attractive cluster is a hypothesis generator, not a finding.
		</p>
	</div>
</div>

<style>
	.methods {
		max-width: 1300px;
		margin: auto;
		padding: 30px 32px 60px;
	}
	header {
		margin-bottom: 28px;
	}
	.eyebrow {
		color: var(--accent);
	}
	h1 {
		font-size: 24px;
		font-weight: 550;
		letter-spacing: -0.5px;
		margin: 8px 0 10px;
	}
	header p {
		color: var(--muted);
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 14px;
	}
	article {
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 7px;
		padding: 20px;
		color: var(--accent);
	}
	h2 {
		font-size: 14px;
		font-weight: 550;
		color: var(--ink);
		margin: 12px 0;
	}
	p {
		font-size: 12px;
		line-height: 1.8;
		color: var(--muted);
	}
	code {
		display: block;
		font: 12px var(--mono);
		padding: 10px;
		background: var(--bg);
		border-radius: 4px;
		color: var(--ink);
	}
	.next {
		margin-top: 24px;
		padding: 22px;
		border: 1px solid var(--line);
		border-left: 2px solid var(--accent);
		border-radius: 5px;
	}
	.next p {
		max-width: 950px;
	}
	@media (max-width: 1000px) {
		.grid {
			grid-template-columns: repeat(2, 1fr);
		}
	}
	@media (max-width: 650px) {
		.methods {
			padding: 22px 14px;
		}
		.grid {
			grid-template-columns: 1fr;
		}
	}
</style>
