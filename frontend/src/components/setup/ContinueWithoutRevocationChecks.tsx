/**
 * The opt-out offered when an environment build fails because Windows
 * could not check a certificate's revocation status.
 *
 * One component for both surfaces that can hit it (the first-run wizard
 * and model preparation), so the wording and the trade-off are stated
 * once and cannot drift apart.
 *
 * Clicking it records the choice on the backend and then retries. The
 * record is a file in the user's AddaxAI folder, so every later
 * environment build honours it too and the user is not asked again.
 */

import { useState } from "react";
import { ExternalLink, Loader2, ShieldOff } from "lucide-react";
import { Button } from "../ui/button";
import { setupApi } from "../../api/setup";

/** The help page section explaining both fixes and how to undo this one. */
export const REVOCATION_HELP_URL =
  "https://docs.addaxai.com/docs/help/locked-down-computers#certificate-errors-during-setup";

interface ContinueWithoutRevocationChecksProps {
  /** Restart the build. Runs after the choice is recorded. */
  onRetry: () => void;
}

export function ContinueWithoutRevocationChecks({
  onRetry,
}: ContinueWithoutRevocationChecksProps) {
  const [saving, setSaving] = useState(false);
  const [failed, setFailed] = useState(false);

  const handleClick = async () => {
    setSaving(true);
    setFailed(false);
    try {
      await setupApi.allowNoRevocationCheck();
      onRetry();
    } catch {
      // Writing the marker is the whole point, so a failure here must
      // not silently start a build that fails the same way again.
      setFailed(true);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        Ask your IT team to exempt the AddaxAI download servers from
        traffic inspection. If that is not possible, you can continue
        without this check. Certificates are still verified, only the
        question of whether one has been revoked is skipped.{" "}
        <a
          href={REVOCATION_HELP_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary underline hover:opacity-80 inline-flex items-center gap-1"
        >
          Read more
          <ExternalLink className="h-3 w-3" />
        </a>
      </p>
      <Button
        variant="outline"
        className="w-full gap-2"
        disabled={saving}
        onClick={handleClick}
      >
        {saving ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Starting...
          </>
        ) : (
          <>
            <ShieldOff className="h-4 w-4" />
            Continue without revocation checks
          </>
        )}
      </Button>
      {failed && (
        <p className="text-xs text-destructive">
          Could not save that choice. Please try again.
        </p>
      )}
    </div>
  );
}
