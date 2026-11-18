/**
 * Bottom-right gradient legend for the observation rate map.
 *
 * Uses a Leaflet Control (not a React component) so it participates
 * in Leaflet's control stack. The legend label mirrors the dashboard
 * metric: observations in the user's rate unit.
 *
 * The numbers are the colours' own scale: the ramp ends at the
 * domain's cap, not at the maximum, so the top label is the cap, with
 * a "+" when a site lies above it.
 */

import { useEffect } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";

import { rampGradient, type RateRamp, type RateScaleDomain } from "../../lib/heat-color-scale";
import { formatRate, rateFromPer100, rateLabel } from "../../lib/trap-night-rate";

/** A legend tick: the domain is per 100 trap nights, like the backend. */
const tick = (ratePer100: number) => formatRate(rateFromPer100(ratePer100));

interface MapLegendProps {
  domain: RateScaleDomain;
  ramp: RateRamp;
}

export function MapLegend({ domain, ramp }: MapLegendProps) {
  const map = useMap();

  useEffect(() => {
    const legend = new L.Control({ position: "bottomright" });

    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "info legend");
      // Theme tokens, not literals: the legend is raw Leaflet DOM, so
      // it does not inherit Tailwind classes and must reference the
      // CSS variables itself to follow dark mode.
      div.style.backgroundColor = "hsl(var(--card))";
      div.style.color = "hsl(var(--card-foreground))";
      div.style.border = "1px solid hsl(var(--border))";
      div.style.padding = "10px";
      div.style.borderRadius = "4px";
      div.style.boxShadow = "0 2px 4px rgba(0,0,0,0.1)";

      const top = tick(domain.cap) + (domain.max > domain.cap ? "+" : "");

      div.innerHTML = `
        <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; line-height: 1.3;">
          Observations<br>${rateLabel()}
        </div>
        <div style="display: flex; align-items: center;">
          <div style="
            width: 20px;
            height: 150px;
            background: ${rampGradient(ramp, "to top")};
            border: 1px solid hsl(var(--border));
            border-radius: 2px;
            margin-right: 8px;
          "></div>
          <div style="
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 150px;
            font-size: 11px;
          ">
            <div>${top}</div>
            <div>${tick(domain.cap / 2)}</div>
            <div>0</div>
          </div>
        </div>
      `;

      return div;
    };

    legend.addTo(map);
    return () => {
      legend.remove();
    };
  }, [map, domain, ramp]);

  return null;
}
