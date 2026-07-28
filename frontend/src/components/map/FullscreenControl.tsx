/**
 * Top-right button that toggles the map container into fullscreen.
 *
 * Uses a Leaflet control (not a React component) so the button lives
 * in the native Leaflet control stack. Adds `.map-fullscreen` to the
 * container element; the class is defined in the map's CSS file.
 * Escape key exits fullscreen.
 */

import { useEffect, useRef } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";

const EXPAND_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

const COLLAPSE_ICON = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;

export function FullscreenControl() {
  const map = useMap();
  const isFullscreenRef = useRef(false);
  const btnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const control = new L.Control({ position: "topright" });

    control.onAdd = () => {
      const container = L.DomUtil.create("div", "leaflet-bar");
      const btn = L.DomUtil.create("button", "", container) as HTMLButtonElement;
      btn.type = "button";
      btn.title = "Toggle fullscreen";
      btn.innerHTML = EXPAND_ICON;
      btn.style.cssText =
        "display:flex;align-items:center;justify-content:center;width:34px;height:34px;background:white;border:none;cursor:pointer;";

      L.DomEvent.disableClickPropagation(container);

      btn.addEventListener("click", () => {
        const mapContainer = map.getContainer();
        isFullscreenRef.current = !isFullscreenRef.current;

        if (isFullscreenRef.current) {
          mapContainer.classList.add("map-fullscreen");
          btn.innerHTML = COLLAPSE_ICON;
        } else {
          mapContainer.classList.remove("map-fullscreen");
          btn.innerHTML = EXPAND_ICON;
        }

        map.invalidateSize();
      });

      btnRef.current = btn;
      return container;
    };

    control.addTo(map);
    return () => {
      control.remove();
    };
  }, [map]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreenRef.current) {
        isFullscreenRef.current = false;
        map.getContainer().classList.remove("map-fullscreen");
        if (btnRef.current) btnRef.current.innerHTML = EXPAND_ICON;
        map.invalidateSize();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [map]);

  return null;
}
