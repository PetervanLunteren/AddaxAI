/**
 * About page.
 *
 * App-level page (not project-scoped). Reachable from the global
 * hamburger menu. Reads the version via the Electron IPC; in dev /
 * browser the version falls back to "(dev)".
 *
 * The contributors row pulls from GitHub's public contributors API at
 * page-open time. If GitHub is unreachable or rate-limited, the
 * component degrades silently — the surrounding copy still makes sense
 * without avatars.
 */

import { useEffect, useState } from "react";
import { HomeButton } from "../components/layout/HomeButton";
import { useQuery } from "@tanstack/react-query";
import { Tag } from "lucide-react";
import { formatVersion } from "@/lib/version";

const REPO = "PetervanLunteren/AddaxAI";
const LICENSE_URL = `https://github.com/${REPO}?tab=MIT-1-ov-file#readme`;

interface GithubContributor {
  login: string;
  id: number;
  avatar_url: string;
  html_url: string;
  contributions: number;
  type: string;
}

export default function AboutPage() {
  const [version, setVersion] = useState<string>("(dev)");

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI?.getVersion) {
      window.electronAPI.getVersion().then(setVersion).catch(() => {
        setVersion("(unknown)");
      });
    }
  }, []);

  const { data: contributors } = useQuery({
    queryKey: ["contributors", REPO],
    queryFn: async (): Promise<GithubContributor[]> => {
      const res = await fetch(
        `https://api.github.com/repos/${REPO}/contributors?per_page=30`,
      );
      if (!res.ok) {
        throw new Error(`GitHub returned ${res.status}`);
      }
      const all: GithubContributor[] = await res.json();
      // Drop obvious bots (dependabot, github-actions[bot], etc.).
      return all.filter(
        (c) => c.type !== "Bot" && !c.login.toLowerCase().includes("[bot]"),
      );
    },
    staleTime: 60 * 60 * 1000, // 1 hour
    retry: false,
  });

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <HomeButton />
            <img
              src="/branding/logo-mark.png"
              alt="AddaxAI"
              className="h-12 w-12 shrink-0"
            />
            <div>
              <h1 className="text-2xl font-bold tracking-tight">About</h1>
              <span className="mt-1 inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs font-mono">
                <Tag className="h-3.5 w-3.5" />
                {formatVersion(version)}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">What is AddaxAI</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            AddaxAI is an open-source project that makes camera trap
            image analysis accessible to all conservationists, with no
            paywalls. The app is released under the MIT license, giving
            you full freedom to use, modify, and share it. Your data
            stays on your machine. Your verification work remains
            private. You stay in complete control of what gets analysed
            and where the results go. AddaxAI also functions as a model
            hub, where developers can share and host classification
            models for others to use, at no cost. The aim is simple:
            help ecologists spend more time on meaningful work, and
            less time on repetitive tasks.
          </p>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Created by</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Built and maintained by Peter van Lunteren (
            <a
              href="https://addaxdatascience.com"
              className="text-primary hover:underline"
            >
              Addax Data Science
            </a>
            ,{" "}
            <a
              href="mailto:peter@addaxdatascience.com"
              className="text-primary hover:underline"
            >
              peter@addaxdatascience.com
            </a>
            ). None of it would look the way it does without Dan
            Morris, who has been a key collaborator and a generous
            adviser on all the difficult stuff.
          </p>

          {contributors && contributors.length > 0 && (
            <div className="mt-4 rounded-lg border bg-zinc-50 p-4">
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">
                Code contributors
              </div>
              <div className="flex flex-wrap gap-2">
                {contributors.map((c) => (
                  <a
                    key={c.id}
                    href={c.html_url}
                    title={`${c.login} · ${c.contributions} commit${c.contributions === 1 ? "" : "s"}`}
                    className="block rounded-full ring-2 ring-transparent hover:ring-primary transition-colors"
                  >
                    <img
                      src={c.avatar_url}
                      alt={c.login}
                      className="h-9 w-9 rounded-full"
                      loading="lazy"
                      // Hide the avatar (and its anchor) instead of
                      // showing a broken-image icon if the GitHub
                      // CDN is unreachable.
                      onError={(e) => {
                        const a = e.currentTarget.closest("a");
                        if (a) a.style.display = "none";
                      }}
                    />
                  </a>
                ))}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Pulled live from GitHub. Order is by commit count, not
                contribution size.
              </p>
            </div>
          )}
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Source and license</h2>
          <div className="mt-2 text-sm text-muted-foreground space-y-2">
            <div>
              AddaxAI source code:{" "}
              <a
                href={`https://github.com/${REPO}`}
                className="text-primary hover:underline"
              >
                github.com/{REPO}
              </a>
            </div>
            <div>
              AddaxAI license:{" "}
              <a
                href={LICENSE_URL}
                className="text-primary hover:underline"
              >
                MIT
              </a>
            </div>
            <p>
              AddaxAI also ships with detection, classification, and
              embedding models from various developers. These models
              are not all created or owned by AddaxAI: each one has its
              own developer, license, citation, and intended use. You
              are responsible for using each model in line with its
              license. Open the{" "}
              <span className="font-medium">Model details</span> link
              below each model in the project settings for the full
              information.
            </p>
          </div>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Citation</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            If AddaxAI was useful in a publication, please cite it as:
          </p>
          {/* Hanging indent for bibliography-style citation: the
              first line sits flush, wrapped lines indent under the
              citation body. */}
          <p className="mt-3 pl-[1.75em] -indent-[1.75em] text-sm text-muted-foreground">
            van Lunteren, P., (2023). AddaxAI: A no-code platform to
            train and deploy custom YOLOv5 object detection models.{" "}
            <em>Journal of Open Source Software</em>, 8(88), 5581,{" "}
            <a
              href="https://doi.org/10.21105/joss.05581"
              className="text-primary hover:underline"
            >
              https://doi.org/10.21105/joss.05581
            </a>
          </p>
          <p className="mt-3 text-sm text-muted-foreground">
            Citations for the individual models you used (MegaDetector,
            DINOv2, classification models, etc.) are available via the{" "}
            <span className="font-medium">Model details</span> link below
            each model in the project settings.
          </p>
        </section>

      </main>
    </div>
  );
}
