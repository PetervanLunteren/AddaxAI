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
// polyfill self-detects the missing-flag platforms and registers an
// @font-face for a flags-only Twemoji webfont. Two non-default
// arguments are critical:
//
// 1. The font name has to also appear in the body's font-family stack
//    (handled in index.css). The polyfill registers the @font-face
//    but doesn't apply it anywhere; without the CSS prefix the
//    unicode-range never kicks in and the polyfill silently no-ops.
//
// 2. We pass a local /TwemojiCountryFlags.woff2 path so the font
//    loads from the bundled frontend instead of cdn.jsdelivr.net.
//    Camera-trap testers often run with no internet and the CDN
//    fetch would fail silently, leaving flags broken even with the
//    polyfill installed. The woff2 is ~77 kB and lives in public/.
polyfillCountryFlagEmojis("Twemoji Country Flags", "/TwemojiCountryFlags.woff2")

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
