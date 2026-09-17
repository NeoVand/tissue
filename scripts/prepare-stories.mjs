/** Package the attributed, bounded TinyStories corpus already prepared by Pattern.
 * Usage: node scripts/prepare-stories.mjs /path/to/pattern/static/data/tinystories
 * This copies source data; it does not generate substitute stories.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const source = process.argv[2];
if (!source)
	throw new Error('Pass the directory containing corpus.json, tokens.bin and LICENSE.html.');
const destination = resolve('static/data/tinystories');
const hash = (data) => createHash('sha256').update(data).digest('hex');
const [metadataBytes, tokens, license] = await Promise.all(
	['corpus.json', 'tokens.bin', 'LICENSE.html'].map((file) => readFile(resolve(source, file)))
);
const metadata = JSON.parse(metadataBytes);
if (
	metadata.dataset !== 'TinyStories' ||
	metadata.chars.length !== 96 ||
	metadata.trainTokens + metadata.validationTokens !== tokens.length ||
	tokens.some((id) => id >= metadata.chars.length)
)
	throw new Error('Source corpus failed its token/vocabulary contract.');
const decode = (bytes) => Array.from(bytes, (id) => metadata.chars[id]).join('');
const training = decode(tokens.subarray(0, metadata.trainTokens));
const validation = decode(tokens.subarray(metadata.trainTokens)).split('\n\n').filter(Boolean);
if (validation.length !== 184 || validation.some((story) => story.length < 129))
	throw new Error('Expected 184 complete validation stories of at least 129 characters.');
if (
	new Set(validation).size !== validation.length ||
	validation.some((story) => training.includes(story))
)
	throw new Error('An exact held-out story duplicate was found.');
const provenance = {
	version: 1,
	id: `tinystories-ascii-v1:${hash(tokens)}`,
	preparedAt: new Date().toISOString(),
	source: metadata.source,
	authors: metadata.authors,
	license: metadata.license,
	licenseUrl: metadata.licenseUrl,
	preparation:
		'Copied byte-for-byte from the bounded TinyStories subset prepared by Pattern. Original punctuation was normalized to printable ASCII; this is character-level language modeling.',
	sourcePreparationDate: metadata.prepared,
	files: {
		'corpus.json': { bytes: metadataBytes.length, sha256: hash(metadataBytes) },
		'tokens.bin': { bytes: tokens.length, sha256: hash(tokens) },
		'LICENSE.html': { bytes: license.length, sha256: hash(license) }
	},
	split: {
		training:
			'Original training subset only; 2,097 original stories, 1,773,785 characters. Windows may cross training story boundaries.',
		calibration:
			'First 32 complete stories of the original validation subset; only these stories define probe geometry.',
		evaluation:
			'Remaining 152 complete validation stories; fixed evaluation windows never enter training or the geometry.',
		trainingBlankLineSegments: training.split('\n\n').filter(Boolean).length,
		note: 'Three original training stories contain internal blank lines, so blank-line segments are not the original training-story count.',
		exactHeldOutDuplicatesInTraining: 0,
		validationStoriesUnique: true
	},
	limitations:
		'This is a bounded character-level subset with 128-character context, not the full TinyStories benchmark or a pretrained general-purpose language model.'
};
await mkdir(destination, { recursive: true });
await Promise.all([
	writeFile(resolve(destination, 'corpus.json'), metadataBytes),
	writeFile(resolve(destination, 'tokens.bin'), tokens),
	writeFile(resolve(destination, 'LICENSE.html'), license),
	writeFile(resolve(destination, 'provenance.json'), JSON.stringify(provenance, null, 2) + '\n')
]);
console.log(JSON.stringify({ destination, ...provenance }, null, 2));
