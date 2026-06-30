/**
 * Headless bridge between the native Electron application menu and the
 * React app. The menu is built in the Electron main process; renderer-backed
 * items send a string command over the "menu:command" channel and this
 * component runs the matching action (navigation, one of the app dialogs, a
 * folder-open, the diagnostic export, or the species-name toggle).
 *
 * This is the successor to the old AppHamburger dropdown: same actions, same
 * dialogs, no button UI. It renders nothing except the four dialogs it owns.
 * Mounted once at the app root (next to DownloadCompleteToasts in App.tsx),
 * inside the router and query provider so navigation and queries work.
 *
 * It only does anything in Electron, where window.electronAPI is defined. In
 * a plain browser there is no native menu, so these actions are unavailable.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { setupApi } from "../../api/setup";
import { backupApi } from "../../api/backup";
import { diagnosticsApi } from "../../api/diagnostics";
import { getSpeciesNameMode, setSpeciesNameMode } from "../../lib/species-name-mode";
import { ResetAppDialog } from "../diagnostics/ResetAppDialog";
import { BackupNowDialog } from "../diagnostics/BackupNowDialog";
import { RestoreBackupDialog } from "../diagnostics/RestoreBackupDialog";
import { CheckForUpdatesDialog } from "../diagnostics/CheckForUpdatesDialog";

type DialogId = "reset" | "updates" | "backup" | "restore" | null;

const FALLBACK_VERSION = "(dev)";

export function MenuCommands() {
  const navigate = useNavigate();
  const [dialog, setDialog] = useState<DialogId>(null);
  const [version, setVersion] = useState<string>(FALLBACK_VERSION);

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

    const exportDiagnostic = async () => {
      try {
        await diagnosticsApi.downloadDiagnosticZip();
        toast.success("Diagnostic report saved to Downloads");
      } catch (err) {
        toast.error(`Could not build diagnostic report: ${(err as Error).message}`);
      }
    };

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
        case "open-user-data":
          void openUserDataFolder();
          break;
        case "open-backups":
          void openBackupsFolder();
          break;
        case "export-diagnostic":
          void exportDiagnostic();
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
    </>
  );
}
