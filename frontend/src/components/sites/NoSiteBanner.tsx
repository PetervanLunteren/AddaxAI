/**
 * Banner for the Map and Export pages when some of a project's
 * deployments have no site assigned (`site_id IS NULL`). These can't
 * be plotted or exported because we don't know where they are.
 *
 * Distinct from MissingCoordsBanner: that one nudges the user to add
 * GPS to a known site (e.g. "North ridge"). This one nudges them to
 * pick a site at all. Both can render simultaneously.
 */

import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

import { buttonVariants } from "../ui/button";
import { Callout } from "../ui/callout";

interface NoSiteBannerProps {
  projectId: string;
  /** How many deployments in this project have no site assigned. */
  count: number;
  /** Verb phrase to complete the headline. Defaults to "on the map". */
  context?: string;
}

export function NoSiteBanner({
  projectId,
  count,
  context = "on the map",
}: NoSiteBannerProps) {
  if (count === 0) return null;

  const depLabel = count === 1 ? "deployment" : "deployments";
  const verb = count === 1 ? "isn't" : "aren't";
  const has = count === 1 ? "has" : "have";

  return (
    <Callout
      variant="warning"
      action={
        <Link
          to={`/projects/${projectId}/deployments?site=missing`}
          className={buttonVariants({ variant: "outline", size: "sm" }) + " bg-white"}
        >
          <ArrowRight />
          Assign a site
        </Link>
      }
    >
      <span className="font-medium">
        {count} {depLabel} {verb} {context}.
      </span>{" "}
      {count === 1 ? "It" : "They"} {has} no site assigned.
    </Callout>
  );
}
