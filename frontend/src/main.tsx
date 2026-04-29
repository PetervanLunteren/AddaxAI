import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { polyfillCountryFlagEmojis } from 'country-flag-emoji-polyfill'
import 'leaflet/dist/leaflet.css'
import 'react-day-picker/style.css'
import './index.css'
import App from './App.tsx'

// Windows' Segoe UI Emoji ships without country flag glyphs by design,
// so 🇪🇺 / 🇳🇱 / etc. render as the letter pair (e.g. "EU"). The
// polyfill self-detects the missing-flag platforms, loads a tiny
// flags-only Twemoji webfont, and prepends it to the body font-family
// so flag codepoints are rendered from there. All non-flag emoji
// keep using the system emoji font, so the cactus and globe look
// the same as before.
polyfillCountryFlagEmojis()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
