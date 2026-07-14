import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

import { Button } from "../ui/button";

/**
 * Always-visible "go home" control for the top-level screens (folder-run,
 * projects overview, About). Drawn at rest, so it's discoverable without
 * hovering (unlike a bare clickable logo). Shared so all three surfaces
 * stay identical. Mirrors the app's icon-button idiom: a ghost Button in a
 * Link.
 */
export function HomeButton() {
  return (
    <Link to="/" aria-label="Home" title="Home" className="shrink-0">
      <Button variant="ghost" size="icon">
        <ArrowLeft className="h-4 w-4" />
      </Button>
    </Link>
  );
}
