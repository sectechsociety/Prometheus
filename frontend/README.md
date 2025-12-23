# Prometheus Frontend 🎨

Modern React + Vite frontend for the Prometheus prompt enhancement system.

## 📋 Overview

A responsive, feature-rich web interface for enhancing prompts across different AI models (ChatGPT, Claude, Gemini). Built with React 18, Vite for fast development, and custom CSS with dark/light theme support.

## 🚀 Features

- **Multi-Model Support**: Switch between ChatGPT, Claude, and Gemini
- **Real-Time Enhancement**: Generate 1-5 prompt variations instantly
- **Copy/Export Tools**: 
  - Individual copy buttons per result
  - Copy all prompts at once
  - Export as formatted TXT
  - Export as structured JSON
- **Character Counter**: 2000-character limit with visual warnings
- **Theme Toggle**: Beautiful dark/light mode with smooth transitions
- **API Health Monitoring**: Real-time backend status indicator
- **Responsive Design**: Works on desktop, tablet, and mobile

## 📁 Project Structure

```
frontend/
├── src/
│   ├── main.jsx              # Application entry point
│   ├── App.jsx               # Main app component with theme logic
│   ├── components/
│   │   ├── PromptBar.jsx     # Input textarea with character counter
│   │   ├── Results.jsx       # Results container with export actions
│   │   └── ResultCard.jsx    # Individual result card with copy button
│   ├── api/
│   │   └── augment.js        # API client for backend communication
│   └── styles/
│       └── index.css         # Global styles with CSS variables
├── index.html                # HTML template
├── package.json              # Dependencies and scripts
├── vite.config.mjs           # Vite configuration with proxy
├── Dockerfile                # Production container image
└── README.md                 # This file
```

## 🛠️ Tech Stack

- **Framework**: React 18
- **Build Tool**: Vite 5.x (lightning-fast HMR)
- **Styling**: Custom CSS with CSS variables for theming
- **API Client**: Fetch API with async/await
- **Icons**: Emoji-based (no external icon library)
- **Fonts**: BBH Sans Hegarty (custom), Montserrat (Google Fonts)

## 📦 Dependencies

```json
{
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "@vitejs/plugin-react": "^4.3.4",
  "vite": "^5.4.11"
}
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on port 8000 (see `backend/README.md`)

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
# Create optimized production build
npm run build

# Preview production build locally
npm run preview
```

## 🔧 Configuration

### API Endpoint

The frontend connects to the backend via proxy configuration in `vite.config.mjs`:

```javascript
export default {
  server: {
    proxy: {
      '/augment': 'http://localhost:8000',
      '/health': 'http://localhost:8000'
    }
  }
}
```

To change the backend URL, update the proxy settings or modify `src/api/augment.js`.

### Environment Variables

- `VITE_API_BASE_URL`: Override backend URL (default: uses proxy)

## 🎨 Theming

The app supports dark and light themes using CSS custom properties:

```css
:root {
  --bg: #0a0a0f;
  --text: #e0e0f0;
  --accent: #7c3aed;
  --card: #1a1a2e;
  /* ... more variables */
}

[data-theme="light"] {
  --bg: #f5f5f7;
  --text: #1a1a2e;
  /* ... light theme overrides */
}
```

Toggle theme with the button in the top-right corner.

## 📡 API Integration

### Backend Endpoints Used

1. **POST /augment**
  - Request: `{ raw_prompt, num_variations }`
  - Response: `{ enhanced_prompts, detected_prompt_type, original_prompt }`

2. **GET /health**
   - Response: `{ status, model_loaded, version }`

### Example Usage

```javascript
import { augment } from './api/augment';

const result = await augment("Write a function to calculate fibonacci", 3);

console.log(result.enhanced_prompts); // Array of 3 enhanced prompts
```

## 🧪 Testing

```bash
# Manual testing checklist
1. Enter a prompt and submit
2. Verify 3 enhanced prompts appear
3. Test copy button on each result
4. Test "Copy All" functionality
5. Test export as TXT and JSON
6. Test character counter (try >2000 chars)
7. Toggle light/dark theme
8. Check API health indicator
```

## 🐳 Docker Deployment

### Development Mode

```bash
# From project root
docker-compose up frontend
```

### Production Mode

```bash
# Build image
docker build -t prometheus-frontend .

# Run container
docker run -p 5173:5173 prometheus-frontend
```

## 🎯 Component Guide

### PromptBar.jsx

Input component with character counter and validation.

**Props**: None (manages own state)

**Features**:
- 2000 character limit
- Yellow warning at 1800 chars (90%)
- Red error at 2000 chars (100%)
- Blocks submission when over limit
- Configurable variations (1-5)

### Results.jsx

Container for displaying enhanced prompts with bulk actions.

**Props**: 
- `results`: Array of enhanced prompts
- `loading`: Boolean for loading state

**Features**:
- Export as TXT (formatted with dividers)
- Export as JSON (includes metadata)
- Copy all prompts to clipboard

### ResultCard.jsx

Individual result display with copy functionality.

**Props**:
- `prompt`: String - The enhanced prompt text
- `index`: Number - Card index (1-based display)

**Features**:
- Copy to clipboard with visual feedback
- Fallback for older browsers
- 2-second "Copied!" confirmation

## 🔥 Performance

- **Bundle Size**: ~150KB gzipped
- **First Load**: <1s (with backend running)
- **Hot Module Replacement**: <100ms
- **Build Time**: ~5s

## 🐛 Troubleshooting

### Port 5173 already in use

```bash
# Kill existing process
pkill -f vite

# Or use different port
vite --port 3000
```

### API connection failed

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check proxy configuration in `vite.config.mjs`
3. Check browser console for CORS errors

### Theme not persisting

Theme state is currently in-memory. To persist, add to `App.jsx`:

```javascript
localStorage.setItem('theme', theme);
// On mount:
const savedTheme = localStorage.getItem('theme') || 'dark';
```

## 🚀 Future Enhancements

- [ ] User authentication
- [ ] Prompt history/favorites
- [ ] Customizable templates
- [ ] A/B testing prompts
- [ ] Analytics dashboard
- [ ] PWA support (offline mode)
- [ ] Keyboard shortcuts
- [ ] Drag-and-drop file upload

## 📄 License

Part of Project Prometheus. See main LICENSE file.

## 🤝 Contributing

This is a production-ready application. For feature requests or bug reports, please create an issue in the main repository.

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: November 16, 2025
