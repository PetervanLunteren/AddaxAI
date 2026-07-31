/**
 * Headless bridge between the native Electron application menu and the
 * React app. The menu is built in the Electron main process; renderer-backed
 * items send a string command over the "menu:command" channel and this
 * component runs the matching action (navigation, one of the app dialogs, a
 * folder-open, the diagnostic export, or the species-name toggle).
 *
 * This is the successor to the old AppHamburger dropdown: same actions, same
 * dialogs, no button UI. It renders nothing except the dialogs it owns.
 * Mounted once at the app root (next to DownloadCompleteToasts in App.tsx),
 * inside the router and query provider so navigation and queries work.
 *
 * Menu commands only arrive in Electron, where window.electronAPI is defined.
 * The one exception is the legacy-install prompt, which opens by itself when
 * an old AddaxAI is found, and so also fires in the browser dev server.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { setupApi } from "../../api/setup";
import { backupApi } from "../../api/backup";
import { exportDiagnosticReport } from "../../lib/diagnostic-export";
import { getSpeciesNameMode, setSpeciesNameMode } from "../../lib/species-name-mode";
import { ResetAppDialog } from "../diagnostics/ResetAppDialog";
import { BackupNowDialog } from "../diagnostics/BackupNowDialog";
import { RestoreBackupDialog } from "../diagnostics/RestoreBackupDialog";
import { CheckForUpdatesDialog } from "../diagnostics/CheckForUpdatesDialog";
import { RemoveLegacyDialog } from "../diagnostics/RemoveLegacyDialog";

type DialogId = "reset" | "updates" | "backup" | "restore" | "legacy" | null;

const FALLBACK_VERSION = "(dev)";

// Set when the user ticks "don't ask me again" in the legacy-install
// prompt. A UI preference, so it lives with the other one in
// localStorage rather than becoming another marker file in ~/AddaxAI.
const LEGACY_PROMPT_DISMISSED = "addaxai.legacy-prompt-dismissed";

export function MenuCommands() {
  const navigate = useNavigate();
  const [dialog, setDialog] = useState<DialogId>(null);
  const [version, setVersion] = useState<string>(FALLBACK_VERSION);
  const [legacyDismissed, setLegacyDismissed] = useState(
    () => localStorage.getItem(LEGACY_PROMPT_DISMISSED) === "true",
  );

  // Cached once; "Open user data folder" needs an absolute path that
  // varies per OS.
  const { data: setupStatus } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    staleTime: Infinity,
  });

  // App version for the Check-for-updates dialog.
  useEffect(() => {
    window.electronAPI
      ?.getVersion?.()
      .then(setVersion)
      .catch(() => setVersion("(unknown)"));
  }, []);

  // Dev-only: these dialogs are normally opened from the Electron native
  // menu, which doesn't exist in the browser dev server. Open one straight
  // from the URL hash so it can be previewed on localhost, e.g.
  // http://localhost:5173/#restore (also #backup, #reset, #updates, #legacy).
  // Tree-shaken out of production builds (import.meta.env.DEV is false).
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const openFromHash = () => {
      const h = window.location.hash.replace("#", "");
      if (
        h === "restore" ||
        h === "backup" ||
        h === "reset" ||
        h === "updates" ||
        h === "legacy"
      ) {
        setDialog(h as DialogId);
      }
    };
    openFromHash();
    window.addEventListener("hashchange", openFromHash);
    return () => window.removeEventListener("hashchange", openFromHash);
  }, []);

  // Legacy AddaxAI 6 still on the machine? Ask once per launch, unless
  // the user ticked "don't ask me again". Gated on setup being ready so
  // it never lands on top of the first-run wizard. The scan is a couple
  // of exists() calls, so running it every launch costs nothing.
  const { data: legacy } = useQuery({
    queryKey: ["legacy-install"],
    queryFn: setupApi.getLegacyInstall,
    enabled: Boolean(setupStatus?.ready) && !legacyDismissed,
    staleTime: Infinity,
  });

  // A ref rather than dependency identity, so a background refetch can
  // never pop the dialog back up after the user closed it.
  const legacyPromptShown = useRef(false);

  useEffect(() => {
    if (legacyPromptShown.current || legacyDismissed || !legacy?.found) return;
    legacyPromptShown.current = true;
    setDialog("legacy");
  }, [legacy?.found, legacyDismissed]);

  // Reflect the stored species-name mode in the menu radio. Sent on mount;
  // a change reloads the page, so the next mount re-syncs the checkmark.
  useEffect(() => {
    window.electronAPI?.setSpeciesNameMenuMode?.(getSpeciesNameMode());
  }, []);

  // Gate the setup-only menu items on the wizard finishing. The query
  // shares its cache key with SetupGate (which polls), so `ready` flips to
  // true once setup completes and this re-runs to enable the items.
  useEffect(() => {
    window.electronAPI?.setMenuSetupReady?.(Boolean(setupStatus?.ready));
  }, [setupStatus?.ready]);

  // Dispatch menu commands. Re-subscribes when setupStatus loads so the
  // folder-open handler has the resolved path.
  useEffect(() => {
    const api = window.electronAPI;
    if (!api?.onMenuCommand) return;

    const openUserDataFolder = async () => {
      if (!setupStatus?.user_data_dir) {
        toast.error("User data path is unknown.");
        return;
      }
      const err = await api.openPath(setupStatus.user_data_dir);
      if (err) toast.error(`Could not open folder: ${err}`);
    };

    const openBackupsFolder = async () => {
      try {
        const { path } = await backupApi.getDir();
        const err = await api.openPath(path);
        if (err) toast.error(`Could not open folder: ${err}`);
      } catch (err) {
        toast.error(`Could not locate backups folder: ${(err as Error).message}`);
      }
    };

    return api.onMenuCommand((id) => {
      switch (id) {
        case "nav-home":
          navigate("/");
          break;
        case "new-project":
          // ?new=1 tells ProjectsPage to open the create dialog on arrival.
          navigate("/projects?new=1");
          break;
        case "new-folder-run":
          navigate("/folder-runs/new");
          break;
        case "about":
          navigate("/about");
          break;
        case "check-updates":
          setDialog("updates");
          break;
        case "backup":
          setDialog("backup");
          break;
        case "restore":
          setDialog("restore");
          break;
        case "reset":
          setDialog("reset");
          break;
        case "remove-legacy":
          setDialog("legacy");
          break;
        case "open-user-data":
          void openUserDataFolder();
          break;
        case "open-backups":
          void openBackupsFolder();
          break;
        case "export-diagnostic":
          void exportDiagnosticReport();
          break;
        case "species-common":
          setSpeciesNameMode("common");
          break;
        case "species-scientific":
          setSpeciesNameMode("scientific");
          break;
      }
    });
  }, [navigate, setupStatus]);

  return (
    <>
      <ResetAppDialog
        open={dialog === "reset"}
        onOpenChange={(o) => setDialog(o ? "reset" : null)}
      />
      <BackupNowDialog
        open={dialog === "backup"}
        onOpenChange={(o) => setDialog(o ? "backup" : null)}
      />
      <RestoreBackupDialog
        open={dialog === "restore"}
        onOpenChange={(o) => setDialog(o ? "restore" : null)}
      />
      <CheckForUpdatesDialog
        open={dialog === "updates"}
        onOpenChange={(o) => setDialog(o ? "updates" : null)}
        currentVersion={version}
      />
      <RemoveLegacyDialog
        open={dialog === "legacy"}
        onOpenChange={(o) => setDialog(o ? "legacy" : null)}
        dontAskAgain={legacyDismissed}
        onDontAskAgainChange={(value) => {
          localStorage.setItem(LEGACY_PROMPT_DISMISSED, String(value));
          setLegacyDismissed(value);
        }}
      />
    </>
  );
}
