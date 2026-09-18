import type { Locator, Page } from '@playwright/test';
export async function tools(lab: Locator | Page, panel: 'model' | 'inspector') {
	await lab
		.getByRole('button', { name: panel === 'model' ? 'Model & runs' : 'Inspect', exact: true })
		.click();
}
export async function disclose(lab: Locator | Page, text: string) {
	const summary = lab.locator('summary').filter({ hasText: text }).first();
	const open = await summary.evaluate(
		(element) => (element.parentElement as HTMLDetailsElement).open
	);
	if (!open) await summary.click();
}
