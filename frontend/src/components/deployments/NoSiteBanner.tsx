/**
 * Banner that warns about deployments with no camera site.
 *
 * Rendered by GPS-dependent pages (Map, Dashboard sun-time activity,
 * Activity overlap sun-time, Export when CamtrapDP / GeoJSON are
 * selected) whenever the user's current filter set contains
 * deployments that will be silently excluded because they have no
 * latitude / longitude to plot or record.
 *
 * The "View deployments" button deep-links to the deployments page
 * with the synthetic "(no site)" filter pre-applied, so users can
 * immediately see which deployments triggered the banner.
 */

import { useNavigate } from "react-router-dom";
import { Button } from "../ui/button";
import { Callout } from "../ui/callout";
import { NO_SITE_SENTINEL } from "../../lib/filter-url";

interface NoSiteBannerProps {
  projectId: string;
  /**
   * Count of deployments with no camera site. When `message` is not
   * provided, the banner uses this to auto-format
   *   "{count} deployment(s) have no camera site. {reason}"
   * and hides itself when count is 0.
   */
  count?: number;
  /**
   * One sentence describing what is missing on *this* page because of
   * the null-site deployments. Keep it short and direct (no
   * exclamation marks). Example:
   *   "They are not shown on the map."
   *   "They are skipped from CamtrapDP and GeoJSON exports."
   */
  reason?: string;
  /**
   * Full banner text, used verbatim when the count-based framing is
   * not a good fit (e.g. "no camera sites have GPS at all" rather than
   * "N deployments have no site"). When provided, `count` and `reason`
   * are ignored and the banner renders unconditionally.
   */
  message?: string;
}

export function NoSiteBanner({
  projectId,
  count,
  reason,
  message,
}: NoSiteBannerProps) {
  const navigate = useNavigate();

  let label: string;
  if (message !== undefined) {
    label = message;
  } else {
    if ((count ?? 0) <= 0) return null;
    label = `${count} deployment${count === 1 ? "" : "s"} have no camera site. ${reason ?? ""}`.trim();
  }

  return (
    <Callout
      variant="info"
      className="mb-4"
      action={
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() =>
            navigate(
              `/projects/${projectId}/deployments?site_ids=${NO_SITE_SENTINEL}`,
            )
          }
        >
          View deployments
        </Button>
      }
    >
      {label}
    </Callout>
  );
}
