import { Callout } from "@/components/ui/callout";

/**
 * Standing notice shown wherever a classification model is picked and "none"
 * is currently selected. Sets the expectation that a detection-only run will
 * not name species. Info, not warning: detector-only is a valid mode and the
 * first-run default, so this informs without nagging.
 *
 * `noneAvailable` is the case where the chosen data type has no
 * classification model at all (underwater today). The picker is then left
 * out and this notice stands in its place, so it says why rather than
 * implying a choice was made.
 */
export function NoClassifierNotice({ noneAvailable = false }: { noneAvailable?: boolean }) {
  return (
    <Callout variant="info" size="compact">
      {noneAvailable
        ? "There is no classification model for this data type yet, so AddaxAI " +
          "finds the animals but does not identify the species. You can label " +
          "them yourself in the Labels section."
        : "Without a classification model, AddaxAI detects animals but does not " +
          "identify the species. You can label them yourself in the Labels section."}
    </Callout>
  );
}
