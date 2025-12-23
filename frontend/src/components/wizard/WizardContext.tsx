/**
 * Wizard Context - manages wizard state across steps.
 *
 * Following DEVELOPERS.md principles:
 * - Type hints everywhere
 * - Simple, clear structure
 * - Explicit state management
 */

import { createContext, useContext, useState, ReactNode } from "react";

interface WizardContextType {
  currentStep: number;
  totalSteps: number;
  setTotalSteps: (total: number) => void;
  nextStep: () => void;
  prevStep: () => void;
  goToStep: (step: number) => void;
  canGoNext: boolean;
  canGoPrev: boolean;
  setCanGoNext: (can: boolean) => void;
}

const WizardContext = createContext<WizardContextType | undefined>(undefined);

interface WizardProviderProps {
  children: ReactNode;
  onComplete?: () => void;
}

export function WizardProvider({ children, onComplete }: WizardProviderProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [canGoNext, setCanGoNext] = useState(true);

  const nextStep = () => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep(currentStep + 1);
    } else if (currentStep === totalSteps - 1 && onComplete) {
      onComplete();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const goToStep = (step: number) => {
    if (step >= 0 && step < totalSteps) {
      setCurrentStep(step);
    }
  };

  const canGoPrev = currentStep > 0;

  return (
    <WizardContext.Provider
      value={{
        currentStep,
        totalSteps,
        setTotalSteps,
        nextStep,
        prevStep,
        goToStep,
        canGoNext,
        canGoPrev,
        setCanGoNext,
      }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const context = useContext(WizardContext);
  if (!context) {
    throw new Error("useWizard must be used within a WizardProvider");
  }
  return context;
}
