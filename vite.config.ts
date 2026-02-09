import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mkcert from 'vite-plugin-mkcert'

export default defineConfig(({ mode }) => {
  const isProd = mode === 'production'
  return {
    // ✅ 关键：让产物引用相对路径 ./assets/...
    base: isProd ? './' : '/',
    server: {
      host: true,
      https: !isProd,
      port: 5173
    },
    plugins: [react(), ...(isProd ? [] : [mkcert()])]
  }
})

