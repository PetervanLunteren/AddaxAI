import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { polyfillCountryFlagEmojis } from 'country-flag-emoji-polyfill'
import 'leaflet/dist/leaflet.css'
import 'react-day-picker/style.css'
import './index.css'
import App from './App.tsx'
import { logger } from './lib/logger'

// Windows' Segoe UI Emoji ships without country flag glyphs by design,
// so 🇪🇺 / 🇳🇱 / etc. render as the letter pair (e.g. "EU"). The
// polyfill self-detects the missing-flag platforms, loads a tiny
// flags-only Twemoji webfont, and prepends it to the body font-family
// so flag codepoints are rendered from there. All non-flag emoji
// keep using the system emoji font, so the cactus and globe look
// the same as before.
polyfillCountryFlagEmojis()

// Catch escaped errors at the global level so they end up in
// backend.log instead of dying in DevTools where users never look.
// Both handlers are best-effort: failures forwarding to the logger
// must never themselves throw, so we swallow secondary errors.
window.addEventListener('error', (event) => {
  try {
    logger.error('window.error: ' + (event.message || 'unknown'), {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      stack: event.error?.stack,
    })
  } catch {
    /* ignore */
  }
})

window.addEventListener('unhandledrejection', (event) => {
  try {
    const reason = event.reason
    const message =
      reason instanceof Error
        ? reason.message
        : typeof reason === 'string'
          ? reason
          : JSON.stringify(reason)
    logger.error('unhandledrejection: ' + message, {
      stack: reason instanceof Error ? reason.stack : undefined,
    })
  } catch {
    /* ignore */
  }
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
