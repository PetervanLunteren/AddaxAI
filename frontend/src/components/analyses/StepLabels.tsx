/**
 * Step 4: Labels - Expected label selection with modern styling.
 */

import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Callout } from "@/components/ui/callout";
import { PawPrint, Plus, Info } from "lucide-react";

interface StepLabelsProps {
  labelsList: string[];
  onLabelsChange: (labels: string[]) => void;
}

export function StepLabels({ labelsList, onLabelsChange }: StepLabelsProps) {
  const handleOpenLabelSelector = () => {
    // TODO: Open label selection modal
  };

  return (
    <div className="space-y-6">
      <div>
        <Label className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <PawPrint className="w-5 h-5 text-blue-600" />
          Label Presence
        </Label>
        <p className="text-sm text-gray-600 mt-2">
          Specify which labels are expected in your project area. This helps improve classification accuracy (optional).
        </p>
      </div>

      <div className="space-y-4">
        <Button
          type="button"
          variant="outline"
          onClick={handleOpenLabelSelector}
          className="w-full h-16 border-2 border-dashed hover:border-blue-500 hover:bg-blue-50 transition-colors group"
        >
          <div className="flex items-center gap-3">
            <Plus className="w-5 h-5 text-gray-400 group-hover:text-blue-600 transition-colors" />
            <span className="font-medium">Select expected labels</span>
          </div>
        </Button>

        {labelsList.length > 0 && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg animate-in fade-in slide-in-from-top-2 duration-300">
            <p className="text-sm font-medium text-green-900 mb-3">
              Selected labels ({labelsList.length}):
            </p>
            <div className="flex flex-wrap gap-2">
              {labelsList.map((label) => (
                <span
                  key={label}
                  className="px-3 py-1.5 bg-white border border-green-300 rounded-full text-sm text-green-800 font-medium"
                >
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}

        {labelsList.length === 0 && (
          <Callout variant="info" title="Label selection is optional">
            <p>
              You can add label information later, or let the classifier identify all possible labels automatically.
            </p>
          </Callout>
        )}
      </div>
    </div>
  );
}
