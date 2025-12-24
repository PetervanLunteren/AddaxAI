/**
 * Deployment Wizard - 3-step wizard for creating deployment queue entries.
 *
 * Steps:
 * 1. Data - folder selection
 * 2. Deployment - site selection
 * 3. Species - expected species (optional)
 *
 * Note: Detection and classification models are now configured at the project level.
 */

import { useState } from "react";
import { Wizard, WizardStep } from "@/components/wizard/Wizard";
import { StepData } from "./StepData";
import { StepDeployment } from "./StepDeployment";
import { StepSpecies } from "./StepSpecies";
import { useAddToQueue } from "@/hooks/useDeploymentQueue";

interface DeploymentWizardProps {
  projectId: string;
}

export function DeploymentWizard({ projectId }: DeploymentWizardProps) {
  // Wizard state
  const [folderPath, setFolderPath] = useState("");
  const [siteId, setSiteId] = useState<string | null>(null);
  const [speciesList, setSpeciesList] = useState<string[]>([]);

  const addToQueue = useAddToQueue();

  const resetWizard = () => {
    setFolderPath("");
    setSiteId(null);
    setSpeciesList([]);
  };

  const handleComplete = async () => {
    try {
      await addToQueue.mutateAsync({
        project_id: projectId,
        folder_path: folderPath,
        site_id: siteId,
        detection_model_id: null, // Models come from project settings
        classification_model_id: null, // Models come from project settings
        species_list: speciesList.length > 0 ? { species: speciesList } : null,
      });

      // Reset wizard state
      resetWizard();

      // TODO: Show success message
      alert("Added to queue successfully!");
    } catch (error) {
      console.error("Failed to add to queue:", error);
      alert("Failed to add to queue. Please try again.");
    }
  };

  return (
    <Wizard
      steps={["Data", "Deployment", "Species"]}
      onComplete={handleComplete}
      onReset={resetWizard}
      submitLabel="Add to Queue"
      isSubmitting={addToQueue.isPending}
    >
      <WizardStep stepIndex={0}>
        <StepData folderPath={folderPath} onFolderChange={setFolderPath} />
      </WizardStep>

      <WizardStep stepIndex={1}>
        <StepDeployment
          projectId={projectId}
          siteId={siteId}
          onSiteChange={setSiteId}
        />
      </WizardStep>

      <WizardStep stepIndex={2}>
        <StepSpecies speciesList={speciesList} onSpeciesChange={setSpeciesList} />
      </WizardStep>
    </Wizard>
  );
}
