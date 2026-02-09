import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(async ({ mode }) => {
  const isProd = mode === 'production'

  const plugins = [react()]

  // ✅ 只在本地开发模式下启用 mkcert
  if (!isProd) {
    const mkcert = (await import('vite-plugin-mkcert')).default
    plugins.push(mkcert())
  }

  return {
    // GitHub Pages 项目部署：使用相对路径
    base: isProd ? './' : '/',

    server: {
      host: true,
      https: !isProd,
      port: 5173
    },

    plugins
  }
})

