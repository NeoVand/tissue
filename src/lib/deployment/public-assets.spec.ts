import { describe, expect, it } from 'vitest';
import { publicAsset } from './public-assets';

describe('deployment-only public asset URLs', () => {
	it.each([
		'/data/tinystories/tokens.bin',
		'/data/tinystories-bpe/tokenizer.json',
		'/experiments/stories-small-seed-42.tissue',
		'/experiments/token-stories-small-seed-42.tissue.part-0'
	])('preserves canonical %s at root and prefixes it only when deployed under a base', (path) => {
		expect(publicAsset(path, '')).toBe(path);
		expect(publicAsset(path, '/')).toBe(path);
		expect(publicAsset(path, '/tissue')).toBe(`/tissue${path}`);
		expect(publicAsset(path, '/tissue/')).toBe(`/tissue${path}`);
	});
	it('does not prefix already adapted, relative, or external URLs', () => {
		for (const path of [
			'/tissue/experiments/index.json',
			'local.json',
			'https://example.com/a',
			'//example.com/a'
		])
			expect(publicAsset(path, '/tissue')).toBe(path);
	});
});
