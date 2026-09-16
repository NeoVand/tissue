import { defineConfig } from '@playwright/test';

export default defineConfig({
	webServer: {
		command: 'pnpm run build && pnpm run preview',
		port: 4173,
		timeout: 180_000,
		reuseExistingServer: !process.env.CI
	},
	timeout: 180_000,
	workers: 1,
	use: {
		baseURL: 'http://localhost:4173',
		launchOptions: { args: ['--enable-unsafe-webgpu'] },
		screenshot: 'only-on-failure',
		trace: 'retain-on-failure'
	},
	testMatch: '**/*.e2e.{ts,js}'
});
