import { Callout } from "@/components/ui/callout";

/**
 * Standing notice shown wherever a classification model is picked and "none"
 * is currently selected. Sets the expectation that a detection-only run will
 * not name species. Info, not warning: detector-only is a valid mode and the
 * first-run default, so this informs without nagging.
 */
export function NoClassifierNotice({
  detectorClassCount = 0,
  detectorModelName,
}: {
  detectorClassCount?: number;
  detectorModelName?: string;
}) {
  if (detectorClassCount > 0) {
    return (
      <Callout variant="info" size="compact">
        The selected detector already labels detections with {detectorClassCount} classes. To include those names and confidence as classification results without another inference pass, choose {detectorModelName ? `“${detectorModelName}”` : "the same detector"} in Classification model.
      </Callout>
    );
  }
  return (
    <Callout variant="info" size="compact">
      Without a classification model, AddaxAI detects animals but does not
      identify the species. You can label them yourself in the Labels section.
    </Callout>
  );
}
