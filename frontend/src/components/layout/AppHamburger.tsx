/**
 * Floating top-right hamburger menu hosting all app-wide actions.
 *
 * Project Settings stays project-scoped; this menu is for things that
 * apply to the whole app: About, Documentation, Open user data folder,
 * Check for updates, Export diagnostic report, Reset application, Quit.
 *
 * Dropdown shape mirrors AddaxAI-Connect's UserMenu so the two apps
 * stay visually consistent for users moving between them.
 */

import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  BookOpen,
  Download,
  FolderOpen,
  Info,
  LogOut,
  Menu,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { setupApi } from "../../api/setup";
import { diagnosticsApi } from "../../api/diagnostics";
import { cn } from "../../lib/utils";
import { ResetAppDialog } from "../diagnostics/ResetAppDialog";
import { CheckForUpdatesDialog } from "../diagnostics/CheckForUpdatesDialog";

const DOCS_URL = "https://github.com/PetervanLunteren/AddaxAI-WebUI";
const FALLBACK_VERSION = "(dev)";

type DialogId = "reset" | "updates" | null;

export function AppHamburger() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [dialog, setDialog] = useState<DialogId>(null);
  const [version, setVersion] = useState<string>(FALLBACK_VERSION);
  const menuRef = useRef<HTMLDivElement>(null);

  // Project pages (/projects/:id/*) hide the hamburger entirely.
  // Each project page renders its own inline <BugReportButton> in
  // the page header, so there's nothing for AppHamburger to add
  // there. The full app menu (About, Reset, Documentation, etc.)
  // is reachable from the projects index, the only place outside
  // a project where AppHamburger renders.
  const isProjectPage = /^\/projects\/[^/]+/.test(location.pathname);

  // The About page is itself reached *from* the hamburger. Showing
  // the menu again on top of it is redundant — the page has its own
  // "Back to projects" button for navigation, and any other app-level
  // action (Reset, Documentation, Quit) is one click away by going
  // back home.
  const isAboutPage = location.pathname.startsWith("/about");
  if (isAboutPage) {
    return null;
  }

  // Cache the user data dir once; the menu's "Open user data folder"
  // entry needs an absolute path that varies per OS.
  const { data: setupStatus } = useQuery({
    queryKey: ["setup-status"],
    queryFn: setupApi.getStatus,
    staleTime: Infinity,
  });

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI?.getVersion) {
      window.electronAPI
        .getVersion()
        .then(setVersion)
        .catch(() => setVersion("(unknown)"));
    }
  }, []);

  // Click-outside-to-close.
  useEffect(() => {
    if (!isOpen) return;
    const handle = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [isOpen]);

  const close = () => setIsOpen(false);

  const exportDiagnostic = async () => {
    close();
    try {
      await diagnosticsApi.downloadDiagnosticZip();
      toast.success("Diagnostic report saved to Downloads");
    } catch (err) {
      toast.error(`Could not build diagnostic report: ${(err as Error).message}`);
    }
  };

  const openUserDataFolder = async () => {
    close();
    if (!setupStatus?.user_data_dir) {
      toast.error("User data path is unknown.");
      return;
    }
    if (!window.electronAPI?.openPath) {
      toast.error(`Path: ${setupStatus.user_data_dir}`);
      return;
    }
    const err = await window.electronAPI.openPath(setupStatus.user_data_dir);
    if (err) toast.error(`Could not open folder: ${err}`);
  };

  const openDocumentation = () => {
    close();
    // External-link clicks go through Electron's setWindowOpenHandler
    // → shell.openExternal, so a plain anchor is enough. We use a
    // synthetic anchor click instead of window.open() for the same
    // hook-handling.
    const a = document.createElement("a");
    a.href = DOCS_URL;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const quitApp = async () => {
    close();
    if (window.electronAPI?.quitApp) {
      await window.electronAPI.quitApp();
    } else {
      toast.message("Close this browser tab to quit (dev mode).");
    }
  };

  if (isProjectPage) {
    return null;
  }

  return (
    <>
      {/* Floats fixed at the right edge of the page-chrome column,
          12px past the action button's right edge.

          The page header is mx-auto max-w-7xl with lg:px-8 padding,
          so the action button's right edge sits at:
            (viewport + 1280) / 2 - 32       on wide screens
            viewport - 32                    on narrow screens (<1280)

          Setting CSS `right` (distance from viewport right edge) to
          `50vw - 41rem` puts the hamburger's left edge ~12px past
          the action button's right edge on wide screens; floored at
          12px so the hamburger never disappears off-screen on narrow
          windows.

          Vertical: the Projects page (the most-trafficked landing
          page) has a 64px row height because of its h-16 logo. Action
          button (h-10, 40px) is centred in that row, so top sits at
          16 (py-4) + (64-40)/2 = 28px. Other pages with the canonical
          52px row will see the hamburger 6px below their action
          button — acceptable mild offset; matching everywhere
          would require either a per-page hook or shrinking the
          Projects logo to h-12. */}
      <div
        ref={menuRef}
        className="fixed top-[28px] z-40"
        style={{ right: "max(0.75rem, calc(50vw - 41.25rem))" }}
      >
        <button
          onClick={() => setIsOpen((v) => !v)}
          className={cn(
            "inline-flex h-10 w-10 items-center justify-center rounded-md border bg-white shadow-sm transition-colors",
            isOpen ? "bg-accent" : "hover:bg-accent",
          )}
          aria-label="App menu"
          aria-expanded={isOpen}
        >
          <Menu className="h-5 w-5" />
        </button>

        {isOpen && (
          <div
            role="menu"
            className="absolute right-0 mt-2 w-64 rounded-md border bg-white shadow-lg"
          >
            <Section>
              <Item
                icon={Info}
                label="About"
                onClick={() => {
                  close();
                  navigate("/about");
                }}
              />
              <Item
                icon={BookOpen}
                label="Documentation"
                onClick={openDocumentation}
              />
            </Section>

            <Separator />

            <Section>
              <Item
                icon={FolderOpen}
                label="Open user data folder"
                onClick={openUserDataFolder}
              />
              <Item
                icon={RefreshCw}
                label="Check for updates"
                onClick={() => {
                  close();
                  setDialog("updates");
                }}
              />
            </Section>

            <Separator />

            <Section>
              <Item
                icon={Download}
                label="Export diagnostic report"
                onClick={exportDiagnostic}
              />
              <Item
                icon={RotateCcw}
                label="Reset application"
                onClick={() => {
                  close();
                  setDialog("reset");
                }}
              />
            </Section>

            <Separator />

            <Section>
              <Item icon={LogOut} label="Quit" onClick={quitApp} />
            </Section>
          </div>
        )}
      </div>

      <ResetAppDialog
        open={dialog === "reset"}
        onOpenChange={(o) => setDialog(o ? "reset" : null)}
      />
      <CheckForUpdatesDialog
        open={dialog === "updates"}
        onOpenChange={(o) => setDialog(o ? "updates" : null)}
        currentVersion={version}
      />
    </>
  );
}

function Section({ children }: { children: React.ReactNode }) {
  return <div className="py-1">{children}</div>;
}

function Separator() {
  return <div className="border-t" />;
}

interface ItemProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
}

function Item({ icon: Icon, label, onClick }: ItemProps) {
  return (
    <button
      role="menuitem"
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors hover:bg-accent"
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}
