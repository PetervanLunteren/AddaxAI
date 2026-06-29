import { Callout } from "@/components/ui/callout";

/**
 * Standing notice shown wherever a classification model is picked and "none"
 * is currently selected. Sets the expectation that a detection-only run will
 * not name species. Info, not warning: detector-only is a valid mode and the
 * first-run default, so this informs without nagging.
 */
export function NoClassifierNotice() {
  return (
    <Callout variant="info" size="compact">
      Without a classification model, AddaxAI detects animals but does not
      identify the species. You can label them yourself in the Labels section.
    </Callout>
  );
}
