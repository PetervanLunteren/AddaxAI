/**
 * Home screen.
 *
 * Task-based chooser: pick what you want AddaxAI to do, not which mode
 * you are in. Two equal-weight paths:
 *
 * 1. Analyse a folder: a quick one-off run, results out, no setup.
 * 2. Build a project: the stored, revisitable project workspace.
 *
 * The two cards differentiate on one-off vs persistent, not on camera
 * count (a folder run handles many cameras too). A folder run hands you
 * the raw output and steps back; a project asks for where each camera
 * was, and uses that location to turn the results into insight (maps,
 * rates, trends).
 *
 * Visual: a full-bleed forest photo (shared with AddaxAI-Connect's
 * login, so the two apps feel related) behind two frosted-glass cards.
 * A scrim under the centered heading and behind the cards keeps text
 * WCAG-legible over the image. Both cards share the same chrome so the
 * user is not nudged toward one by layout alone.
 *
 * Timelapse Analyser is no longer a separate mode: its launcher
 * (`AddaxAI.exe --timelapse <folder>`) now opens "Analyse a folder"
 * with the folder pre-filled, and the folder run's
 * addaxai-recognitions.json is what Timelapse imports.
 */

import { useNavigate } from "react-router-dom";
import { ArrowRight, ExternalLink, LayoutDashboard, FolderOpen } from "lucide-react";
import { buttonVariants } from "../components/ui/button";
import { LogoPlate } from "../components/layout/LogoPlate";
import { cn } from "../lib/utils";

// The guides index: every guide carries its video tutorial at the top,
// and the first two (analyse a folder, build a project) answer the very
// question this screen asks. Same target as Help > Video tutorials.
const TUTORIALS_URL = "https://docs.addaxai.com/docs/category/guides/";

// Frosted-glass surface. backdrop-filter is set inline so the look does
// not depend on the Tailwind backdrop-blur utilities being enabled.
const GLASS: React.CSSProperties = {
  backgroundColor: "rgba(255, 255, 255, 0.10)",
  backdropFilter: "blur(22px) saturate(150%)",
  WebkitBackdropFilter: "blur(22px) saturate(150%)",
};

export function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen overflow-hidden text-white">
      {/* Background photo + scrim. Decorative, so no alt text. */}
      <div
        className="absolute inset-0 scale-105 bg-cover bg-center"
        style={{ backgroundImage: "url('/home-background.webp')" }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(120% 90% at 50% 0%, rgba(10,30,28,0.25), transparent 60%)," +
            "linear-gradient(180deg, rgba(8,22,20,0.55) 0%, rgba(8,22,20,0.30) 35%, rgba(8,22,20,0.66) 100%)",
        }}
      />

      <div className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-7 lg:px-8">
        {/* Centered logo, heading, and the two glass cards. Top-aligned
            (not vertically centred) so the cards never fall off the
            bottom on short screens. */}
        <main className="flex flex-1 flex-col items-center justify-start pt-1 sm:pt-3">
          <div className="animate-in fade-in slide-in-from-bottom-2 duration-500 text-center">
            <LogoPlate className="mx-auto mb-7" logoClassName="h-14" />
            <h1 className="text-4xl font-extrabold tracking-tight drop-shadow-md sm:text-5xl">
              What do you want to do?
            </h1>
            <p className="mt-3 text-lg text-white/80">
              Pick the path that matches your task.
            </p>
          </div>

          <div className="mt-8 grid w-full max-w-4xl gap-6 lg:grid-cols-2">
            <ChoiceCard
              icon={<FolderOpen className="h-6 w-6" />}
              title="Analyse a folder"
              lead="A quick one-off run. Point at a folder, get results, move on."
              body="Run the AI, clean up labels if you want, then get files out: detection tables, a recognition file for Timelapse, species-separated folders, visualised or blurred images. No interpretation, you do your own analysis with the files."
              cta="Start folder analysis"
              onClick={() => navigate("/folder-runs/new")}
            />
            <ChoiceCard
              icon={<LayoutDashboard className="h-6 w-6" />}
              title="Build a project"
              lead="A workspace you come back to and keep adding to."
              body="Add where each camera was, and AddaxAI turns the results into insight: species counts, dashboards and maps, activity, and trap-night rates. Keep a verification history and export to Camtrap DP. Everything stays in the app to revisit and add to."
              cta="Open projects"
              onClick={() => navigate("/projects")}
            />
          </div>

          <p className="mt-6 text-sm text-white/70">
            New here?{" "}
            <a
              href={TUTORIALS_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-white underline underline-offset-2 hover:text-white/85"
            >
              Watch the video tutorials
              <ExternalLink className="h-3 w-3" />
            </a>
          </p>
        </main>
      </div>
    </div>
  );
}

interface ChoiceCardProps {
  icon: React.ReactNode;
  title: string;
  lead: string;
  body: string;
  cta: string;
  onClick: () => void;
}

function ChoiceCard({ icon, title, lead, body, cta, onClick }: ChoiceCardProps) {
  // The whole card is the button, so a click (or Enter / Space) anywhere
  // navigates. The "cta" below is a non-interactive visual cue styled to
  // look like a button; making it a real <button> would nest interactives.
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={title}
      style={GLASS}
      className="group flex cursor-pointer flex-col rounded-[22px] border border-white/20 p-8 text-left
        shadow-[0_20px_50px_-20px_rgba(0,0,0,0.55)] transition-transform duration-200
        hover:-translate-y-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/70
        focus-visible:ring-offset-2 focus-visible:ring-offset-transparent"
    >
      <div
        className="grid place-items-center rounded-2xl border border-white/25 text-white"
        style={{ backgroundColor: "rgba(255,255,255,0.16)", height: "3.25rem", width: "3.25rem" }}
      >
        {icon}
      </div>
      <h2 className="mt-5 text-xl font-bold tracking-tight">{title}</h2>
      <p className="mt-1.5 text-sm text-white/85">{lead}</p>
      <p className="mt-3 flex-1 text-[13.5px] leading-relaxed text-white/70">{body}</p>
      <span
        className={cn(
          buttonVariants({ size: "lg" }),
          "mt-6 self-start bg-[#f4f0e3] text-[#0a4044] shadow-lg transition-transform",
          "group-hover:translate-x-0.5 group-hover:bg-white",
        )}
      >
        {cta}
        <ArrowRight className="h-4 w-4" />
      </span>
    </button>
  );
}
