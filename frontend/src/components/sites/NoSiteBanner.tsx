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
import { AlertCircle, ArrowRight } from "lucide-react";

import { buttonVariants } from "../ui/button";

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
    <div className="flex items-start gap-3 rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
      <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0" />
      <div className="flex-1 space-y-3">
        <p>
          <span className="font-medium">
            {count} {depLabel} {verb} {context}.
          </span>{" "}
          {count === 1 ? "It" : "They"} {has} no site assigned.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Link
            to={`/projects/${projectId}/deployments?site=missing`}
            className={buttonVariants({ variant: "outline", size: "sm" }) + " bg-white"}
          >
            <ArrowRight />
            Assign a site
          </Link>
        </div>
      </div>
    </div>
  );
}
