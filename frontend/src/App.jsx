import React, { useState, useEffect } from 'react'
import './styles/index.css'
import PromptBar from './components/PromptBar'
import Results from './components/Results'
import { augment, checkHealth } from './api/augment'

export default function App(){
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [theme, setTheme] = useState('dark')
  const [metadata, setMetadata] = useState(null)
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Check API health on mount
  useEffect(() => {
    async function checkApiHealth() {
      try {
        const health = await checkHealth()
        setApiStatus(health)
      } catch (err) {
        console.error('API health check failed:', err)
        setApiStatus({ status: 'offline' })
      }
    }
    checkApiHealth()
  }, [])

  function toggleTheme(){
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  async function handleSubmit({ raw, numVariations = 3 }){
    setLoading(true)
    setError(null)
    setMetadata(null)
    
    try{
      const data = await augment(raw, numVariations)
      
      // Set enhanced prompts
      setResults(data.enhanced_prompts || [])
      
      // Store metadata for display
      setMetadata({
        detectedType: data.detected_prompt_type,
        modelType: data.model_type,
        ragUsed: data.rag_context_used,
        ragChunks: data.rag_chunks_count,
        originalPrompt: data.original_prompt
      })
      
    } catch(err){
      setError(err.message || String(err))
      setResults([])
      setMetadata(null)
    } finally{
      setLoading(false)
    }
  }

  return (
    <div className="page-root">
      <header className="site-header">
        <div className="header-content">
          <h1 className="title">Prometheus</h1>
          <p className="tagline">You provide the prompt, we handle the enhancement</p>
          
          {/* API Status Badge */}
          {apiStatus && (
            <div className="api-status">
              <span className={`status-badge status-${apiStatus.status}`}>
                {apiStatus.status === 'healthy' ? '✓ API Connected' : 
                 apiStatus.status === 'degraded' ? '⚠ API Degraded' : 
                 '✗ API Offline'}
              </span>
              {apiStatus.model && (
                <span className="model-badge">
                  {apiStatus.model.type === 'mock' ? '⚡ Prometheus Light v1.0' : '🚀 Prometheus Light v1.0'}
                </span>
              )}
            </div>
          )}
        </div>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </header>

      <main className="center-area">
        <PromptBar onSubmit={handleSubmit} loading={loading} />

        {/* Metadata Display */}
        {metadata && !loading && (
          <div className="metadata-panel">
            <div className="metadata-item">
              <span className="metadata-label">Detected Type:</span>
              <span className="metadata-value">{metadata.detectedType}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Powered by:</span>
              <span className="metadata-value">
                Prometheus Light v1.0 (Pattern-based + RAG)
              </span>
            </div>
            {metadata.ragUsed && (
              <div className="metadata-item">
                <span className="metadata-label">RAG Context:</span>
                <span className="metadata-value">
                  {metadata.ragChunks} guideline{metadata.ragChunks !== 1 ? 's' : ''} retrieved
                </span>
              </div>
            )}
          </div>
        )}

        <section className="results-section">
          {error && <div className="error">❌ {error}</div>}
          <Results items={results} loading={loading} metadata={metadata} />
        </section>
      </main>

      <footer className="site-footer">
        <p>Prometheus • Intelligent Prompt Enhancement</p>
      </footer>
    </div>
  )
}

