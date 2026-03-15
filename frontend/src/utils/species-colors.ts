/**
 * Species color mapping utility
 *
 * Generates consistent colors for species names using a custom gradient.
 * Colors are assigned based on alphabetical order of species names.
 * Gradient: #0f6064 (dark teal) -> #f9f871 (light yellow)
 */
import chroma from 'chroma-js';

// Custom gradient scale from dark teal to light yellow
const speciesScale = chroma.scale(['#0f6064', '#f9f871']);

// Cache for storing species -> color mappings within a context
let speciesOrderCache: Map<string, number> = new Map();

export function setSpeciesContext(speciesList: string[]): void {
  speciesOrderCache.clear();
  const sorted = [...speciesList].map(s => s.toLowerCase()).sort();
  sorted.forEach((species, index) => {
    const position = sorted.length > 1 ? index / (sorted.length - 1) : 0.5;
    speciesOrderCache.set(species, position);
  });
}

export function getSpeciesColor(species: string): string {
  const position = speciesOrderCache.get(species.toLowerCase()) ?? 0.5;
  return speciesScale(position).hex();
}

export function getSpeciesColors(speciesList: string[]): string[] {
  setSpeciesContext(speciesList);
  return speciesList.map(species => getSpeciesColor(species));
}

export function getSpeciesColorWithAlpha(species: string, alpha: number = 0.8): string {
  const position = speciesOrderCache.get(species.toLowerCase()) ?? 0.5;
  return speciesScale(position).alpha(alpha).css();
}

export function getSpeciesChartColors(species: string, backgroundAlpha: number = 0.8): {
  borderColor: string;
  backgroundColor: string;
} {
  return {
    borderColor: getSpeciesColor(species),
    backgroundColor: getSpeciesColorWithAlpha(species, backgroundAlpha),
  };
}

export function getSpeciesTextColor(species: string): string {
  const position = speciesOrderCache.get(species.toLowerCase()) ?? 0.5;
  const bgColor = speciesScale(position);
  return chroma.contrast(bgColor, 'white') >= 3 ? 'white' : '#1f2937';
}
