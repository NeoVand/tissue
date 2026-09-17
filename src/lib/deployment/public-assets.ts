/** Deployment location is separate from the canonical paths saved with scientific evidence. */
export function publicAsset(
	path: string,
	deploymentBase = import.meta.env?.BASE_URL ?? ''
): string {
	const base = deploymentBase.replace(/\/$/, '');
	if (!base || !path.startsWith('/') || path.startsWith('//')) return path;
	if (path === base || path.startsWith(`${base}/`)) return path;
	return `${base}${path}`;
}
