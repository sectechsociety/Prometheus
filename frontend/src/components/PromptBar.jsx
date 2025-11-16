import React, { useState } from 'react'

export default function PromptBar({ onSubmit, loading }){
  const [raw, setRaw] = useState('')
  const [model, setModel] = useState('ChatGPT')
  const [numVariations, setNumVariations] = useState(3)
  
  const MAX_CHARS = 2000
  const charCount = raw.length
  const isOverLimit = charCount > MAX_CHARS
  const warningThreshold = MAX_CHARS * 0.9 // 90%

  function handleSubmit(e){
    e.preventDefault()
    if (!raw.trim() || isOverLimit) return
    onSubmit({ raw, model, numVariations })
  }

  return (
    <form className="prompt-bar" onSubmit={handleSubmit}>
      <div className="textarea-wrapper">
        <textarea
          className="prompt-input"
          placeholder="Type a prompt (e.g. 'Explain how DNS works')"
          value={raw}
          onChange={e=>setRaw(e.target.value)}
          rows={4}
          disabled={loading}
          maxLength={MAX_CHARS}
          onKeyDown={(e)=>{
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e)
            }
          }}
        />
        <div className={`char-counter ${charCount > warningThreshold ? 'warning' : ''} ${isOverLimit ? 'error' : ''}`}>
          {charCount} / {MAX_CHARS}
        </div>
      </div>

      <div className="prompt-controls">
        <select 
          className="model-select" 
          value={model} 
          onChange={e=>setModel(e.target.value)}
          disabled={loading}
          style={{fontFamily: 'Montserrat, sans-serif'}}
        >
          <option value="ChatGPT" style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>ChatGPT</option>
          <option value="Claude" style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>Claude</option>
          <option value="Gemini" style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>Gemini</option>
        </select>

        <select 
          className="variations-select" 
          value={numVariations} 
          onChange={e=>setNumVariations(Number(e.target.value))}
          disabled={loading}
          title="Number of enhanced variations"
          style={{fontFamily: 'Montserrat, sans-serif'}}
        >
          <option value={1} style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>1 variation</option>
          <option value={2} style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>2 variations</option>
          <option value={3} style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>3 variations</option>
          <option value={4} style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>4 variations</option>
          <option value={5} style={{fontFamily: 'Montserrat, sans-serif', fontWeight: 500}}>5 variations</option>
        </select>

        <button className="submit-btn" type="submit" disabled={loading || !raw.trim() || isOverLimit}>
          {loading ? '✨ Enhancing...' : '👾 Enhance'}
        </button>
      </div>
    </form>
  )
}

