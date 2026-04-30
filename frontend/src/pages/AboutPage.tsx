/**
 * About page.
 *
 * App-level page (not project-scoped). Reachable from the global
 * hamburger menu. Reads the version via the Electron IPC; in dev /
 * browser the version falls back to "(dev)".
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "../components/ui/button";

export default function AboutPage() {
  const [version, setVersion] = useState<string>("(dev)");

  useEffect(() => {
    if (typeof window !== "undefined" && window.electronAPI?.getVersion) {
      window.electronAPI.getVersion().then(setVersion).catch(() => {
        setVersion("(unknown)");
      });
    }
  }, []);

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">About</h1>
              <p className="text-sm text-muted-foreground">
                AddaxAI v{version}
              </p>
            </div>
            <Link to="/projects">
              <Button variant="outline" size="sm">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to projects
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">What is AddaxAI</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            AddaxAI is a desktop application for analysing camera trap
            images and videos with AI models. It runs fully offline once
            installed: detection, classification, verification, and
            export all happen on your machine. No cloud, no upload, no
            account.
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            The goal is to make modern camera-trap AI accessible to
            ecologists, conservation NGOs, and reserve managers without
            assuming reliable internet, a cloud subscription, or a
            programming background.
          </p>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Created by</h2>
          <div className="mt-2 text-sm text-muted-foreground space-y-1">
            <div>
              Peter van Lunteren ·{" "}
              <a
                href="https://addaxdatascience.com"
                className="text-primary hover:underline"
              >
                Addax Data Science
              </a>
            </div>
            <div>
              <a
                href="mailto:peter@addaxdatascience.com"
                className="text-primary hover:underline"
              >
                peter@addaxdatascience.com
              </a>
            </div>
          </div>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Source and license</h2>
          <div className="mt-2 text-sm text-muted-foreground space-y-1">
            <div>
              Source code:{" "}
              <a
                href="https://github.com/PetervanLunteren/AddaxAI-WebUI"
                className="text-primary hover:underline"
              >
                github.com/PetervanLunteren/AddaxAI-WebUI
              </a>
            </div>
            <div>License: MIT</div>
          </div>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">Citation</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            If AddaxAI was useful in a publication, please cite it as:
          </p>
          <pre className="mt-2 rounded bg-muted p-3 text-xs font-mono whitespace-pre-wrap">
            van Lunteren, P. ({new Date().getFullYear()}). AddaxAI: a
            desktop platform for AI-assisted camera trap analysis.
            https://github.com/PetervanLunteren/AddaxAI-WebUI
          </pre>
        </section>

        <section className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold tracking-tight">
            Acknowledgements
          </h2>
          <p className="mt-2 text-sm text-muted-foreground">
            AddaxAI builds on the work of many people and projects:
          </p>
          <ul className="mt-2 text-sm text-muted-foreground list-disc list-inside space-y-1">
            <li>
              <strong>MegaDetector</strong> by Dan Morris (object
              detection backbone for animals, people, vehicles).
            </li>
            <li>
              <strong>DINOv2</strong> by Meta AI / FAIR (self-supervised
              visual features for similarity and clustering).
            </li>
            <li>
              <strong>DeepFaune</strong> by CNRS and partners (European
              species classification).
            </li>
            <li>
              <strong>SpeciesNet</strong> by the SpeciesNet team (global
              species classification).
            </li>
            <li>
              Regional model partners: SDZWA, ADS, NEP, BB, JAP, HEX,
              and others listed under each model in the catalogue.
            </li>
            <li>
              Open-source frameworks: Electron, FastAPI, React, Vite,
              SQLAlchemy, HuggingFace Hub, micromamba.
            </li>
          </ul>
        </section>
      </main>
    </div>
  );
}
