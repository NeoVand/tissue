/** UI pacing only. These gates never describe GPU execution time. */
export class LayerPlaybackClock {
	private paused = false;
	private stopped = true;
	private layerSteps = 0;
	private finishCurrentToken = false;
	private milliseconds = 160;
	private epoch = 0;
	private wake: (() => void) | null = null;

	start(paused = false): void {
		this.stop();
		this.stopped = false;
		this.paused = paused;
		this.layerSteps = 0;
		this.finishCurrentToken = false;
	}
	setPace(milliseconds: number): void {
		this.milliseconds = Math.max(16, milliseconds);
		this.wake?.();
	}
	pause(): void {
		this.paused = true;
		this.wake?.();
	}
	play(): void {
		this.paused = false;
		this.layerSteps = 0;
		this.finishCurrentToken = false;
		this.wake?.();
	}
	nextLayer(): void {
		this.paused = true;
		this.layerSteps++;
		this.wake?.();
	}
	nextToken(): void {
		this.paused = true;
		this.finishCurrentToken = true;
		this.wake?.();
	}
	finishToken(): void {
		this.finishCurrentToken = false;
	}
	async waitForFrame(): Promise<boolean> {
		const epoch = this.epoch;
		while (
			epoch === this.epoch &&
			!this.stopped &&
			this.paused &&
			!this.finishCurrentToken &&
			this.layerSteps === 0
		)
			await this.wait();
		if (this.stopped || epoch !== this.epoch) return false;
		if (this.paused && !this.finishCurrentToken && this.layerSteps > 0) this.layerSteps--;
		return true;
	}
	stop(): void {
		this.epoch++;
		this.stopped = true;
		this.wake?.();
	}
	private wait(milliseconds?: number): Promise<void> {
		return new Promise((resolve) => {
			let timer: ReturnType<typeof setTimeout> | undefined;
			const finish = () => {
				if (timer !== undefined) clearTimeout(timer);
				if (this.wake === finish) this.wake = null;
				resolve();
			};
			this.wake = finish;
			if (milliseconds !== undefined) timer = setTimeout(finish, milliseconds);
		});
	}
	async waitLayer(): Promise<boolean> {
		const epoch = this.epoch;
		while (!this.stopped && epoch === this.epoch) {
			if (this.finishCurrentToken) {
				await this.wait(32);
				return !this.stopped && epoch === this.epoch;
			}
			if (this.layerSteps > 0) {
				this.layerSteps--;
				return true;
			}
			if (this.paused) await this.wait();
			else {
				await this.wait(this.milliseconds);
				if (!this.paused && !this.stopped && epoch === this.epoch) return true;
			}
		}
		return false;
	}
}
