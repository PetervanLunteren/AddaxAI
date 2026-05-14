/**
 * Step 4 stub: Review results.
 *
 * The real implementation deep-links to /projects/:runId/verify with
 * the sidebar hidden and the filter pre-set to low-confidence
 * detections + a random sample of high-confidence ones. The user can
 * skip the review entirely. Stubbed for this slice.
 */

import { StepStub } from "./StepStub";

export function FolderRunReviewStep() {
  return (
    <StepStub
      title="Review results"
      description="Check a subset of results before saving final outputs."
      comingNext="The review grid opens with low-confidence detections
plus a random sample of high-confidence ones already selected. Skip
or continue with the current results when ready."
      thisStep="review"
      backTo="run"
      nextTo="save"
    />
  );
}
