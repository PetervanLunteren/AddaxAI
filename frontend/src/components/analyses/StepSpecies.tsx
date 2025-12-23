/**
 * Step 4: Species - Expected species selection with modern styling.
 */

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Callout } from "@/components/ui/callout";
import { PawPrint, Plus, Info } from "lucide-react";

interface StepSpeciesProps {
  speciesList: string[];
  onSpeciesChange: (species: string[]) => void;
}

export function StepSpecies({ speciesList, onSpeciesChange }: StepSpeciesProps) {
  const handleOpenSpeciesSelector = () => {
    // TODO: Open species selection modal
    console.log("Species selector not implemented yet");
  };

  return (
    <div className="space-y-6">
      <div>
        <Label className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <PawPrint className="w-5 h-5 text-blue-600" />
          Species Presence
        </Label>
        <p className="text-sm text-gray-600 mt-2">
          Specify which species are expected in your project area. This helps improve classification accuracy (optional).
        </p>
      </div>

      <div className="space-y-4">
        <Button
          type="button"
          variant="outline"
          onClick={handleOpenSpeciesSelector}
          className="w-full h-16 border-2 border-dashed hover:border-blue-500 hover:bg-blue-50 transition-colors group"
        >
          <div className="flex items-center gap-3">
            <Plus className="w-5 h-5 text-gray-400 group-hover:text-blue-600 transition-colors" />
            <span className="font-medium">Select Expected Species</span>
          </div>
        </Button>

        {speciesList.length > 0 && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
            <p className="text-sm font-medium text-green-900 mb-3">
              Selected species ({speciesList.length}):
            </p>
            <div className="flex flex-wrap gap-2">
              {speciesList.map((species) => (
                <span
                  key={species}
                  className="px-3 py-1.5 bg-white border border-green-300 rounded-full text-sm text-green-800 font-medium"
                >
                  {species}
                </span>
              ))}
            </div>
          </div>
        )}

        {speciesList.length === 0 && (
          <Callout variant="info" title="Species selection is optional">
            <p>
              You can add species information later, or let the classifier identify all possible species automatically.
            </p>
          </Callout>
        )}
      </div>
    </div>
  );
}
