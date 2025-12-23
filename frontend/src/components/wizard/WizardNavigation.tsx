/**
 * Wizard Navigation - Prev/Next buttons with modern styling.
 */

import { useWizard } from "./WizardContext";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Plus, RotateCcw } from "lucide-react";

interface WizardNavigationProps {
  onSubmit?: () => void;
  onReset?: () => void;
  submitLabel?: string;
  isSubmitting?: boolean;
}

export function WizardNavigation({
  onSubmit,
  onReset,
  submitLabel = "Add to Queue",
  isSubmitting = false,
}: WizardNavigationProps) {
  const { currentStep, totalSteps, nextStep, prevStep, canGoNext, canGoPrev } =
    useWizard();

  const isLastStep = currentStep === totalSteps - 1;

  const handleNext = () => {
    if (isLastStep && onSubmit) {
      onSubmit();
    } else {
      nextStep();
    }
  };

  return (
    <div className="flex items-center justify-between gap-4">
      {/* Left: Start Over button */}
      <div>
        {onReset && (
          <Button
            type="button"
            variant="ghost"
            onClick={onReset}
            disabled={isSubmitting}
            className="text-gray-600 hover:text-gray-900"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Start Over
          </Button>
        )}
      </div>

      {/* Right: Previous and Next buttons */}
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={prevStep}
          disabled={!canGoPrev || isSubmitting}
          className="group"
        >
          <ChevronLeft className="w-4 h-4 mr-1 transition-transform group-hover:-translate-x-1" />
          Previous
        </Button>

        <Button
          type="button"
          onClick={handleNext}
          disabled={!canGoNext || isSubmitting}
          className={`group ${isLastStep ? "bg-green-600 hover:bg-green-700" : ""}`}
        >
          {isSubmitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
              Adding...
            </>
          ) : isLastStep ? (
            <>
              <Plus className="w-4 h-4 mr-1" />
              {submitLabel}
            </>
          ) : (
            <>
              Next
              <ChevronRight className="w-4 h-4 ml-1 transition-transform group-hover:translate-x-1" />
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
