import { StoryGeometryEngine } from '../stories/geometry-engine';
import type { TokenStoryAtlas } from './protocol';
import { geometryAtlas } from './geometry';

/** Same audited numeric method, with a tokenizer-aware public boundary. */
export class TokenStoryGeometryEngine {
	private engine = new StoryGeometryEngine();
	build(atlas: TokenStoryAtlas) {
		return this.engine.build(geometryAtlas(atlas));
	}
	setAtlas(atlas: TokenStoryAtlas) {
		return this.engine.setAtlas(geometryAtlas(atlas));
	}
	neighbors(unit: number, k = 6, layer?: number) {
		return this.engine.neighbors(unit, k, layer);
	}
	reset() {
		this.engine.reset();
	}
	destroy() {
		this.engine.destroy();
	}
	dispose() {
		this.destroy();
	}
}
