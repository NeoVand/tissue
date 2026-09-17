<script lang="ts">
	import type { RunRecord } from '$lib/lab/journal';
	import type { QueryStudyRecord } from '$lib/lab/query-journal';
	import { notebook } from '$lib/lab/notebook';
	import { references } from '$lib/lab/references';
	import LearningCurve from './LearningCurve.svelte';
	import Icon from './Icon.svelte';
	let {
		run,
		runs,
		busy,
		onopen,
		onreference,
		onexport,
		onimport,
		onnote,
		queryStudies = [],
		onquery
	}: {
		run: RunRecord | null;
		runs: RunRecord[];
		busy: boolean;
		onopen: (run: RunRecord) => void;
		onreference: (path: string) => void;
		onexport: (run: RunRecord) => void;
		onimport: () => void;
		onnote: (text: string) => void;
		queryStudies?: QueryStudyRecord[];
		onquery?: (record: QueryStudyRecord) => void;
	} = $props();
	let note = $state('');
	const percent = (value: number | undefined) =>
		value === undefined ? '—' : `${(100 * value).toFixed(1)}%`;
</script>

<div class="journal">
	<header class="page-heading">
		<div>
			<span class="eyebrow">Evidence archive</span>
			<h1>Field journal</h1>
			<p>Hypotheses, measurements, failures, and the next experiment.</p>
		</div>
		<button class="secondary" onclick={onimport} disabled={busy}
			><Icon name="upload" />Import run</button
		>
	</header>
	<div class="journal-grid">
		<section>
			<div class="section-label">
				<Icon name="book" />
				<h2>Research notebook</h2>
				<span>{notebook.length} entries</span>
			</div>
			{#each [...notebook].sort((a, b) => b.id.localeCompare(a.id)) as entry (entry.id)}
				<article class="entry">
					<div class="meta">
						<span class="entry-id">{entry.id}</span><span>{entry.kind}</span><time
							>{entry.date}</time
						>
					</div>
					<h3>{entry.title}</h3>
					<p>{entry.text}</p>
					<div class="lesson">{entry.lesson}</div>
					{#if entry.links}<div class="links">
							{#each entry.links as link (link.href)}<!-- eslint-disable-next-line svelte/no-navigation-without-resolve --><a
									href={link.href}
									target="_blank"
									rel="noreferrer">{link.title}<Icon name="right" size={12} /></a
								>{/each}
						</div>{/if}
				</article>
			{/each}
		</section>
		<section>
			<div class="section-label">
				<Icon name="flask" />
				<h2>Recorded experiments</h2>
				<span>2 references</span>
			</div>
			{#each references as reference (reference.seed)}<article class="reference-card">
					<div class="meta">
						<span class="reference-tag">REPRODUCIBLE</span><span>seed {reference.seed}</span>
					</div>
					<h3>{reference.title}</h3>
					<p>{reference.description}</p>
					<button class="secondary" onclick={() => onreference(reference.path)} disabled={busy}
						>Open measured specimen<Icon name="right" size={13} /></button
					>
				</article>{/each}
			{#if queryStudies.length}<div class="section-label saved-label">
					<Icon name="target" />
					<h2>Paired-query studies</h2>
					<span>{queryStudies.length}</span>
				</div>
				{#each queryStudies as study (study.id)}<article class="run-card">
						<div class="meta">
							<span>{study.source}</span><time
								>{new Date(study.createdAt).toLocaleDateString()}</time
							>
						</div>
						<h3>Query shifts · seed {study.measurement.seed}</h3>
						<p>
							Step {study.measurement.step} · 32 assignment groups · 256 interventions<br
							/>{study.measurement.backend.toUpperCase()} · source checkpoint preserved
						</p>
						<div class="actions">
							<button class="text-button" onclick={() => onquery?.(study)}
								>Inspect paired-query study<Icon name="right" size={12} /></button
							>
						</div>
					</article>{/each}
			{/if}
			<div class="section-label saved-label">
				<Icon name="layers" />
				<h2>Saved runs</h2>
				<span>{runs.length} local</span>
			</div>
			{#each runs as record (record.id)}<article class="run-card">
					<div class="meta">
						<span>{record.id === run?.id ? 'ACTIVE RUN' : 'SAVED RUN'}</span><time
							>{new Date(record.createdAt).toLocaleDateString()}</time
						>
					</div>
					<h3>{record.title}</h3>
					<div class="run-stats">
						<span><b>{record.metrics.at(-1)?.step ?? 0}</b> steps</span><span
							><b>{percent(record.metrics.at(-1)?.accuracy)}</b> held-out</span
						>
					</div>
					<LearningCurve metrics={record.metrics} selectedStep={record.metrics.at(-1)?.step ?? 0} />
					<div class="actions">
						<button
							class="text-button"
							onclick={() => onopen(record)}
							disabled={busy || !record.checkpoint}
							>Open specimen<Icon name="right" size={12} /></button
						><button
							class="icon-button"
							aria-label={`Export ${record.title}`}
							onclick={() => onexport(record)}><Icon name="download" size={14} /></button
						>
					</div>
				</article>{/each}
		</section>
		<section>
			<div class="section-label">
				<Icon name="activity" />
				<h2>Run observations</h2>
				<span>{run?.observations.length ?? 0}</span>
			</div>
			{#if run}<div class="field-note">
					<label for="field-note">Leave a field note</label><textarea
						id="field-note"
						rows="3"
						bind:value={note}
						placeholder="A hypothesis, surprise, or next test…"></textarea><button
						class="secondary"
						disabled={!note.trim()}
						onclick={() => {
							onnote(note);
							note = '';
						}}>Keep this observation</button
					>
				</div>
				{#each [...run.observations].reverse() as observation (observation.id)}<article
						class="observation"
					>
						<div class="meta">
							<span>STEP {observation.step}</span><span>{observation.kind}</span><time
								>{new Date(observation.time).toLocaleTimeString()}</time
							>
						</div>
						<h3>{observation.title}</h3>
						<p>{observation.detail}</p>
					</article>{/each}{/if}
		</section>
	</div>
</div>

<style>
	.journal {
		max-width: 1800px;
		margin: auto;
		padding: 24px 28px 60px;
	}
	.page-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 24px;
	}
	.page-heading h1 {
		font-size: 24px;
		letter-spacing: -0.6px;
		font-weight: 550;
		margin: 7px 0;
	}
	.page-heading p {
		margin: 0;
		color: var(--muted);
	}
	.eyebrow {
		color: var(--accent);
	}
	.journal-grid {
		display: grid;
		grid-template-columns: minmax(300px, 1.2fr) minmax(240px, 0.9fr) minmax(280px, 1fr);
		gap: 24px;
	}
	.section-label {
		display: flex;
		align-items: center;
		gap: 8px;
		border-bottom: 1px solid var(--line);
		padding-bottom: 12px;
		color: var(--muted);
	}
	.section-label h2 {
		font-size: 12px;
		font-weight: 550;
		color: var(--ink);
		margin: 0;
	}
	.section-label > span {
		margin-left: auto;
		font: 9px var(--mono);
	}
	.entry,
	.observation {
		padding: 18px 0;
		border-bottom: 1px solid var(--line);
	}
	.meta {
		display: flex;
		align-items: center;
		gap: 9px;
		font: 9px var(--mono);
		text-transform: uppercase;
		color: var(--muted);
	}
	.meta time {
		margin-left: auto;
		font-size: 8px;
	}
	.entry-id {
		color: var(--accent);
	}
	h3 {
		font-size: 13px;
		line-height: 1.5;
		font-weight: 550;
		margin: 11px 0 7px;
	}
	p {
		font-size: 11px;
		line-height: 1.75;
		color: var(--muted);
		margin: 0;
	}
	.lesson {
		margin-top: 12px;
		padding: 10px 12px;
		border: 1px solid color-mix(in srgb, var(--accent) 25%, var(--line));
		border-radius: 5px;
		background: var(--surface);
		font-size: 11px;
		line-height: 1.7;
		color: var(--ink);
	}
	.links,
	.actions {
		display: flex;
		align-items: center;
		gap: 16px;
		margin-top: 12px;
	}
	.links a {
		display: flex;
		gap: 5px;
		align-items: center;
		color: var(--accent);
		font-size: 10px;
		text-decoration: none;
	}
	.reference-card,
	.run-card {
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 14px;
		margin-top: 12px;
	}
	.reference-tag {
		color: var(--accent);
	}
	.reference-card button {
		margin-top: 13px;
	}
	.saved-label {
		margin-top: 24px;
	}
	.run-stats {
		display: flex;
		gap: 18px;
		color: var(--muted);
		font: 10px var(--mono);
		margin: 12px 0 8px;
	}
	.run-stats b {
		color: var(--ink);
		font-weight: 500;
	}
	.actions {
		justify-content: space-between;
	}
	.field-note {
		margin-top: 14px;
		padding: 14px;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 6px;
	}
	.field-note label {
		display: block;
		font-size: 11px;
		margin-bottom: 9px;
	}
	.field-note textarea {
		display: block;
		width: 100%;
		margin-bottom: 10px;
	}
	.observation p {
		overflow-wrap: anywhere;
	}
	@media (max-width: 1150px) {
		.journal-grid {
			grid-template-columns: 1.2fr 1fr;
		}
		.journal-grid > section:last-child {
			grid-column: 1/-1;
		}
	}
	@media (max-width: 700px) {
		.journal {
			padding: 20px 14px;
		}
		.journal-grid {
			display: block;
		}
		.journal-grid > section {
			margin-bottom: 28px;
		}
		.page-heading h1 {
			font-size: 21px;
		}
	}
</style>
