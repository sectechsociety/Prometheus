import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const useBackend = process.env.VITE_USE_BACKEND === '1'

function mockAugmentPlugin(){
  return {
    name: 'mock-augment',
    configureServer(server){
      server.middlewares.use((req, res, next) => {
        if (!useBackend && req.method === 'POST' && req.url && req.url.startsWith('/augment')) {
          let body = ''
          req.on('data', (chunk) => (body += chunk))
          req.on('end', () => {
            try{
              const payload = JSON.parse(body || '{}')
              const raw = payload.raw_prompt || ''
              const detectedType = 'mock-type'
              const enhanced = [
                `[ENHANCED for ${detectedType}] ${raw}`,
                `Be specific: ${raw}`,
                `Add format and constraints: ${raw} (JSON, steps, time, audience)`
              ]
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ enhanced_prompts: enhanced, detected_prompt_type: detectedType }))
            }catch(e){
              res.statusCode = 400
              res.end('Invalid JSON')
            }
          })
          return
        }
        next()
      })
    }
  }
}

export default defineConfig({
  plugins: [react(), mockAugmentPlugin()],
  server: {
    port: 5173,
    host: true,
    proxy: useBackend ? {
      '/augment': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
    } : undefined,
  },
})
