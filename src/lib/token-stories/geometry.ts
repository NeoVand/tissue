import type { TokenStoryAtlas } from './protocol';
import type { StoryAtlas } from '../stories/protocol';
import { validateStoryGeometry } from '../stories/validate-geometry';
export type {
	StoryGeometryResult as TokenStoryGeometryResult,
	StoryNeighbor as TokenStoryNeighbor,
	StoryPoint as TokenStoryPoint
} from '../stories/geometry';
import type { StoryGeometryResult } from '../stories/geometry';

/**
 * Geometry uses hidden-channel fingerprints, never vocabulary IDs or logits.
 * Keep the original measured PCA implementation frozen; this type adapter changes
 * no data, including the actual 4096-token architecture and corpus identity.
 */
export function geometryAtlas(atlas: TokenStoryAtlas): StoryAtlas {
	return {
		...atlas,
		examples: atlas.examples.map((text, i) =>
			JSON.stringify({ text, tokens: atlas.tokenIds[i], positions: atlas.tokenPositions[i] })
		)
	} as unknown as StoryAtlas;
}

export function validateTokenStoryGeometry(
	value: unknown,
	atlas: TokenStoryAtlas
): asserts value is StoryGeometryResult {
	validateStoryGeometry(value, geometryAtlas(atlas));
}
