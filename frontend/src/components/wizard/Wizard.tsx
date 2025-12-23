/**
 * Wizard - reusable multi-step wizard component.
 *
 * Modern timeline design with smooth transitions.
 */

import { ReactNode, useEffect } from "react";
import { WizardProvider, useWizard } from "./WizardContext";
import { WizardProgress } from "./WizardProgress";
import { WizardNavigation } from "./WizardNavigation";

interface WizardProps {
  steps: string[];
  children: ReactNode;
  onComplete: () => void;
  onReset?: () => void;
  submitLabel?: string;
  isSubmitting?: boolean;
  icons?: React.ReactNode[];
}

function WizardContent({
  steps,
  children,
  onComplete,
  onReset,
  submitLabel,
  isSubmitting,
  icons,
}: WizardProps) {
  const { setTotalSteps } = useWizard();

  useEffect(() => {
    setTotalSteps(steps.length);
  }, [steps.length, setTotalSteps]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
      {/* Timeline Header */}
      <div className="bg-gradient-to-br from-gray-50 to-white px-8 pt-8 pb-6 border-b border-gray-100">
        <WizardProgress steps={steps} icons={icons} />
      </div>

      {/* Step Content */}
      <div className="px-8 py-8 min-h-[400px]">{children}</div>

      {/* Navigation Footer */}
      <div className="px-8 py-6 bg-gray-50 border-t border-gray-100">
        <WizardNavigation
          onSubmit={onComplete}
          onReset={onReset}
          submitLabel={submitLabel}
          isSubmitting={isSubmitting}
        />
      </div>
    </div>
  );
}

export function Wizard(props: WizardProps) {
  return (
    <WizardProvider onComplete={props.onComplete}>
      <WizardContent {...props} />
    </WizardProvider>
  );
}

// Export all wizard components for convenience
export { WizardStep } from "./WizardStep";
export { useWizard } from "./WizardContext";
