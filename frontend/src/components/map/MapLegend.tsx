/**
 * Bottom-right gradient legend for the observation rate map.
 *
 * Uses a Leaflet Control (not a React component) so it participates
 * in Leaflet's control stack. The legend label mirrors the dashboard
 * metric: "Observations per 100 trap nights".
 */

import { useEffect } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";

import type { RateScaleDomain } from "../../lib/heat-color-scale";

interface MapLegendProps {
  domain: RateScaleDomain;
}

export function MapLegend({ domain }: MapLegendProps) {
  const map = useMap();

  useEffect(() => {
    const legend = new L.Control({ position: "bottomright" });

    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "info legend");
      div.style.backgroundColor = "white";
      div.style.padding = "10px";
      div.style.borderRadius = "4px";
      div.style.boxShadow = "0 2px 4px rgba(0,0,0,0.1)";

      const middleValue = domain.max / 2;
      const gradientColors = ["#0f6064", "#f9f871"].join(", ");

      div.innerHTML = `
        <div style="font-size: 12px; font-weight: 600; margin-bottom: 8px; line-height: 1.3;">
          Observations per<br>100 trap nights
        </div>
        <div style="display: flex; align-items: center;">
          <div style="
            width: 20px;
            height: 150px;
            background: linear-gradient(to bottom, ${gradientColors});
            border: 1px solid rgba(0,0,0,0.2);
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
            <div>${Math.round(domain.max)}</div>
            <div>${Math.round(middleValue)}</div>
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
  }, [map, domain]);

  return null;
}
