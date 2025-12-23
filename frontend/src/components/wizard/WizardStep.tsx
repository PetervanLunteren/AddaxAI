/**
 * Wizard Step - individual step wrapper with smooth transitions.
 *
 * Fades in/out with slide animation.
 */

import { ReactNode } from "react";
import { useWizard } from "./WizardContext";

interface WizardStepProps {
  children: ReactNode;
  stepIndex: number;
}

export function WizardStep({ children, stepIndex }: WizardStepProps) {
  const { currentStep } = useWizard();

  if (currentStep !== stepIndex) {
    return null;
  }

  return (
    <div
      className="animate-in fade-in slide-in-from-right-4 duration-300"
      key={stepIndex}
    >
      {children}
    </div>
  );
}
