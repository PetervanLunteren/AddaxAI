/**
 * MapLibre GL vector base layer as a react-leaflet layer.
 *
 * Renders a vector style (OpenFreeMap) through MapLibre GL inside a
 * normal Leaflet map, via @maplibre/maplibre-gl-leaflet. Everything
 * else on the map (markers, hexbins, controls, legends) stays plain
 * Leaflet. Ported from AddaxAI Connect's component of the same name;
 * unlike there it is imported statically, because SiteMap already puts
 * maplibre-gl in the main bundle so lazy-loading would buy nothing.
 *
 * maxZoom matters: Leaflet takes the map's zoom range from its layers,
 * and a fitBounds on a single point with an unbounded range zooms to
 * Infinity and crashes MapLibre's matrix math. Raster tile layers carry
 * a default of 18; this layer declares the same. The vector data stops
 * at z14 and MapLibre overzooms it, so rendering stays crisp.
 *
 * Attribution is ours, never the style's: the plugin hands the string
 * to Leaflet's attribution control, which writes it with innerHTML and
 * sanitises nothing. customAttribution makes the plugin use the string
 * we pass rather than the one it fetched, so the remote value never
 * reaches the DOM.
 */

import {
  createElementObject,
  createLayerComponent,
  type LayerProps,
} from "@react-leaflet/core";
import { MaplibreGL } from "@maplibre/maplibre-gl-leaflet";
import "maplibre-gl/dist/maplibre-gl.css";

interface MapLibreGLLayerProps extends LayerProps {
  /** URL of the MapLibre style JSON. */
  styleUrl: string;
  /** Leaflet zoom limit, same default as a raster TileLayer. */
  maxZoom?: number;
  /** Credit line to show. Required, so it can never fall back to the style's. */
  attribution: string;
}

const MapLibreGLLayer = createLayerComponent<
  InstanceType<typeof MaplibreGL>,
  MapLibreGLLayerProps
>(({ styleUrl, maxZoom = 18, attribution }, ctx) => {
  const layer = new MaplibreGL({
    style: styleUrl,
    maxZoom,
    attributionControl: { customAttribution: attribution },
  });
  return createElementObject(layer, ctx);
});

export default MapLibreGLLayer;
