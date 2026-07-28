/**
 * Wizard Progress - horizontal timeline with all steps visible.
 *
 * Modern, compact design with clickable completed steps.
 */

import { useWizard } from "./WizardContext";
import { Check } from "lucide-react";

interface WizardProgressProps {
  steps: string[];
  icons?: React.ReactNode[];
}

export function WizardProgress({ steps, icons }: WizardProgressProps) {
  const { currentStep, goToStep } = useWizard();

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isClickable = isCompleted;

          return (
            <div key={index} className="flex items-center flex-1">
              {/* Step Circle */}
              <button
                onClick={() => isClickable && goToStep(index)}
                disabled={!isClickable}
                className={`
                  relative flex items-center justify-center w-10 h-10 rounded-full font-semibold text-sm
                  transition-all duration-300 shrink-0
                  ${
                    isCompleted
                      ? "bg-gradient-to-br from-green-500 to-green-600 text-white shadow-md cursor-pointer hover:scale-110"
                      : isCurrent
                      ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white shadow-lg ring-4 ring-blue-100"
                      : "bg-gray-200 text-gray-500"
                  }
                `}
              >
                {isCompleted ? (
                  <Check className="w-5 h-5" />
                ) : icons && icons[index] ? (
                  icons[index]
                ) : (
                  index + 1
                )}
              </button>

              {/* Step Label */}
              <div
                className={`
                  ml-3 mr-4 text-sm font-medium transition-colors duration-300
                  ${
                    isCompleted
                      ? "text-green-700"
                      : isCurrent
                      ? "text-blue-700"
                      : "text-gray-500"
                  }
                `}
              >
                {step}
              </div>

              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div className="flex-1 h-0.5 mx-2">
                  <div
                    className={`
                      h-full transition-all duration-500
                      ${
                        index < currentStep
                          ? "bg-gradient-to-r from-green-500 to-green-400"
                          : "bg-gray-200"
                      }
                    `}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
