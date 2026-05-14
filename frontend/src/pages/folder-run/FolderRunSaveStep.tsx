/**
 * Step 5 stub: Save outputs.
 *
 * The real implementation lands in the postprocess-outputs slice:
 * checkboxes for CSV, recognition JSON, separate-into-folders,
 * visualised images, blur people, crops; an output folder picker
 * (default sibling "AddaxAI results" next to the source folder);
 * progress; a completion screen with the promote-to-research-project
 * button. Stubbed for this slice.
 */

import { StepStub } from "./StepStub";

export function FolderRunSaveStep() {
  return (
    <StepStub
      title="Save outputs"
      description="Choose the files and folders you want AddaxAI to create."
      comingNext="The save step writes a CSV table, a recognition JSON,
optional separated subfolders by label, visualised image copies,
people-blurred copies, and per-detection crops. After it finishes, a
button offers to promote the folder run into a research project."
      thisStep="save"
      backTo="review"
      nextTo={null}
    />
  );
}
