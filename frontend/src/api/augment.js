// API Base URL - defaults to localhost for development
const BASE = (import.meta.env && import.meta.env.VITE_API_BASE) || 'http://localhost:8000'

/**
 * Enhance a raw prompt using the Prometheus API
 * 
 * @param {string} rawPrompt - The original user prompt to enhance
 * @param {string} targetModel - Target AI model (ChatGPT, Claude, or Gemini)
 * @param {number} numVariations - Number of enhanced variations to generate (1-5)
 * @param {number} temperature - Sampling temperature (0.0-1.0, higher = more creative)
 * @param {boolean} useRag - Whether to include RAG context from knowledge base
 * @returns {Promise<Object>} Response with enhanced prompts and metadata
 */
export async function augment(
  rawPrompt, 
  targetModel = 'ChatGPT',
  numVariations = 3,
  temperature = 0.7,
  useRag = true
) {
  const url = `${BASE}/augment`
  
  const payload = {
    raw_prompt: rawPrompt,
    target_model: targetModel,
    num_variations: numVariations,
    temperature: temperature,
    use_rag: useRag
  }
  
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Augment failed: ${res.status} ${text}`)
  }
  
  return res.json()
}

/**
 * Check API health status
 * @returns {Promise<Object>} Health status with model and RAG info
 */
export async function checkHealth() {
  const url = `${BASE}/health`
  const res = await fetch(url)
  
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`)
  }
  
  return res.json()
}

export default { augment, checkHealth }

