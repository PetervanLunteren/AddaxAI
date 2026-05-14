# Workflow recommendation

## Recommendation

Add a task-based home screen with one primary path:

1. **Analyse a folder**
2. **Research projects**
3. **Timelapse integration**

This is not the same as offering “simple mode” and “advanced mode”. The user should not have to decide whether they are a simple or advanced user. They should decide what job they want AddaxAI to do today.

The default path should be **Analyse a folder**. It restores the old AddaxAI mental model: choose a folder, choose what to detect, run the model, check results, save outputs. It should feel like a self-contained run, not like entering a database or project management system.

**Research projects** should keep the current project/deployment/site workflow, but present itself as the right place for longer studies, multiple deployments, verification over time, dashboards, insights, and export workflows.

**Timelapse integration** should be a small third utility path, not hidden in a menu. It is a real use case, but narrower than the other two.

## Why this is better than two equal modes

Two equal modes, especially named **Simple mode** and **Projects mode**, would reduce some confusion, but it creates a new problem: users have to understand the product architecture before doing work.

The home screen should instead answer the user's first question:

> I have camera-trap media. What do I want AddaxAI to do with it?

That gives a cleaner mental model:

- **Analyse a folder**: one folder in, useful files out.
- **Research projects**: build a reusable project with sites, deployments, verification, insights, and exports.
- **Timelapse integration**: create a Timelapse recognition file.

The user can understand these choices without knowing what a database, deployment, or project schema is.

## Home screen design

### Page title

**What do you want to do?**

### Primary card

**Analyse a folder**

Caption:

> Run AI on one folder and save results you can use right away.

Supporting text:

> Best for quick camera-trap batches, legacy AddaxAI-style workflows, folder separation, visualised images, people blurring, and CSV or JSON outputs.

Primary button:

> Start folder analysis

Secondary affordance:

> Recent folder runs

### Secondary card

**Research projects**

Caption:

> Manage sites, deployments, verification, dashboards, insights, and exports.

Supporting text:

> Best for studies with multiple camera locations, repeated imports, metadata, long-term verification, maps, activity plots, performance checks, and Camtrap-DP style exports.

Primary button:

> Open projects

Secondary button:

> New project

### Utility card

**Timelapse integration**

Caption:

> Create a recognition file for Timelapse Analyser.

Supporting text:

> Choose a folder, run recognition, then import the generated file in Timelapse.

Primary button:

> Create Timelapse file

## Recommended naming

### Use these names

**Analyse a folder**

Why:

- It describes the job, not the user's skill level.
- It maps directly to the legacy AddaxAI workflow.
- It avoids implying that the output is lower quality.
- It works for both beginners and experts who just want a fast batch run.

**Research projects**

Why:

- It signals a richer workspace without sounding intimidating.
- It fits the current app: projects, deployments, sites, verification, dashboards, insights.
- It makes the cost clear: this path asks for metadata because it gives better long-term analysis.

**Timelapse integration**

Why:

- It names the external tool directly.
- It avoids pretending this is a general mode.

### Avoid these names

**Simple mode**

Problem:

- It can feel like a beginner label.
- It does not explain the output.
- Users may wonder what they lose by choosing it.

**Advanced mode**

Problem:

- It invites the user to choose based on confidence rather than task.
- It can make the project workflow feel like the “correct” one, even when it is unnecessary.

**Quick analysis**

Problem:

- Better than “simple”, but “quick” can imply approximate or temporary.
- Some folder analyses may run for hours on large backlogs.

**Project workspace**

Problem:

- Usable, but a bit abstract.
- “Research projects” better explains why the structure exists.

## Folder analysis workflow

The folder path should feel like a single guided run with five steps.

### Step 1: choose folder

Title:

**Choose media folder**

Caption:

> Select the folder with images or videos you want to analyse.

Controls:

- Folder picker
- Include subfolders checkbox
- Output location selector

Default output:

> Create an `AddaxAI results` folder next to the selected folder.

Important UX rule:

Do not mention projects, deployments, sites, or database here.

### Step 2: choose AI

Title:

**Choose AI models**

Caption:

> Select what AddaxAI should detect and identify.

Default controls:

- Detection model
- Species identification model
- Country or region filter when the selected model supports it

Advanced disclosure:

**Model settings**

Keep thresholds, batch size, taxonomy rollup, smoothing, and GPU settings behind this disclosure. Most users should not see them on first pass.

### Step 3: run analysis

Title:

**Run analysis**

Caption:

> AddaxAI will scan the folder, run the selected models, and write results to the output folder.

Progress should show plain stages:

- Scanning files
- Detecting animals, people, and vehicles
- Identifying species
- Preparing review files
- Saving results

Avoid implementation words such as database import, deployment analysis, embedding, queue entry, or worker.

### Step 4: review results

Title:

**Review results**

Caption:

> Check a subset of results before saving final outputs.

Recommended review choices:

- Review low-confidence detections
- Review a random sample
- Review selected species
- Review people and vehicles
- Skip review

The default should be a suggested review set, not a blank filter builder.

Suggested default:

> Low-confidence animals and a random sample of high-confidence results.

Button labels:

- Start review
- Skip review
- Continue with current results

Do not block postprocessing because review is incomplete. The legacy code already shows that this warning created friction. Let the user continue with clear wording.

### Step 5: save outputs

Title:

**Save outputs**

Caption:

> Choose the files and folders you want AddaxAI to create.

Output options:

- **Results table**: CSV or XLSX
- **Recognition JSON**: AddaxAI JSON for later use
- **Separate into folders**: copy or move files by label
- **Visualised images**: draw boxes and labels
- **Blur people**: create privacy-safe visualised images
- **Crops**: save detected animals as crops

Use checkboxes with short descriptions. Keep destructive choices explicit.

For file movement:

Label:

> Copy or move files

Options:

- Copy files to output folders
- Move files to output folders

Default:

> Copy files to output folders

Warning when move is selected:

> Moving files changes the original folder. Use copy if you want to keep the source folder unchanged.

Completion screen:

Title:

**Folder analysis complete**

Caption:

> Results were saved to `{output_folder}`.

Buttons:

- Open results folder
- Analyse another folder
- Start a research project from these results

## Research projects workflow

The existing Projects area should stay, but the first screen should explain why it exists.

Page title:

**Research projects**

Caption:

> Use projects when you want to combine multiple deployments, keep verification history, analyse metadata, and build exports over time.

Empty state:

> Create a project for studies with multiple camera locations, deployments, maps, verification, dashboards, and exports.

Primary button:

> New research project

Secondary button:

> Analyse a folder instead

Project creation should keep the current model and settings choices, but the wording should consistently explain the scientific structure:

- Project: the study or dataset.
- Site: the camera location.
- Deployment: one camera run at one site during one time period.
- Analysis: the model run on a deployment folder.

## Relationship between folder analysis and research projects

The two paths should not be isolated forever. The best bridge is:

> Start a research project from these results

Show it only after a folder run completes. Do not ask for project metadata before the user gets value.

Bridge flow:

1. User completes **Analyse a folder**.
2. Completion screen offers **Start a research project from these results**.
3. AddaxAI asks for project name and timezone.
4. AddaxAI imports the run as the first deployment.
5. User can add site metadata later.

This gives the simple workflow a natural upgrade path without forcing every user into project concepts at the start.

## Data model recommendation

For the user, **Analyse a folder** should feel like JSON-first and file-first.

Internally, there are two reasonable options:

### Option A: true JSON-first folder runs

The run writes JSON and output files without creating project records.

Pros:

- Cleanest match to the legacy mental model.
- Easier to explain.
- No hidden project state.

Cons:

- Duplicates logic that already exists in the project pipeline.
- Verification, thumbnails, video best frames, filters, and exports may need separate implementations.
- Higher risk of divergence between folder runs and projects.

### Option B: hidden temporary run record

The run uses the same backend services and database tables internally, but the UI presents it as a folder run. The user never sees project, site, or deployment language unless they convert the run to a research project.

Pros:

- Reuses the current pipeline, verification UI, model setup, logging, diagnostics, and export code.
- Lower implementation risk.
- Easier to convert a folder run into a project later.

Cons:

- The app must hide internal project concepts carefully.
- Cleanup and persistence rules need to be explicit.

### Recommended data model

Use **Option B** first.

Keep the user-facing workflow file-first, but use the existing database-backed pipeline internally. The current app already depends on database state for verification, events, filtering, thumbnails, video best frames, logs, backups, and exports. Rebuilding all of that for a pure JSON mode would create two products inside one app.

The important rule is UX, not storage:

> Folder analysis must never make the user think about projects, sites, deployments, or database state.

Persist folder runs in a small **Recent folder runs** list. Let users delete them from the home screen. The source JSON and output files remain in their chosen output folder.

## Home screen layout

Recommended layout:

- Left column, large: **Analyse a folder**
- Right column, stacked: **Research projects**, **Timelapse integration**
- Bottom row: recent work

Recent work should mix both types but label them clearly:

- Folder run: `{folder_name}`, analysed on `{date}`
- Research project: `{project_name}`, `{n} deployments`
- Timelapse run: `{folder_name}`, recognition file created on `{date}`

Do not make users choose a mode before seeing recent work. Recent work should be openable directly.

## Navigation changes

Recommended top-level routes:

- `/home`
- `/folder-runs/new`
- `/folder-runs/:runId`
- `/projects`
- `/projects/:projectId/...`
- `/timelapse`

The sidebar should only appear inside **Research projects**. Folder analysis should use a focused stepper layout. Timelapse should stay a focused single-purpose page.

## Copy rules

Use task words:

- Analyse
- Review
- Save
- Export
- Open results
- Start project

Avoid architecture words in the folder flow:

- Project
- Deployment
- Site
- Database
- Queue
- Worker
- Embedding
- CRUD

Use architecture words only in **Research projects**, where they are part of the value.

## Feature split

### Analyse a folder should include

- Folder selection
- Model selection
- Basic thresholds behind advanced disclosure
- Run progress
- Review subset
- Results table export
- JSON export
- Folder separation
- Visualised images
- People blurring
- Crops
- Open results folder

### Analyse a folder should not include

- Site management
- Deployment metadata
- Maps
- Trap nights
- Activity overlap
- Performance dashboards
- Camtrap-DP export
- Long-term project settings
- Multi-deployment insights

### Research projects should include

- Everything currently tied to projects
- Sites and deployments
- Verification history
- Dashboard
- Insights
- Maps
- Activity overlap
- Performance views
- Export workflows
- Backups and restore
- Model reprocessing

## First implementation slice

Do not build the entire ideal folder workflow at once. Start with a narrow version that proves the mental model.

Version 1:

1. Add the new home screen.
2. Make **Analyse a folder** a guided wrapper around one folder run.
3. Support model selection, run progress, review, CSV export, JSON export, folder separation, visualised images, and people blurring.
4. Keep **Research projects** as the current Projects page with improved naming.
5. Move **Timelapse integration** to the home screen.

Defer:

- Converting folder runs into research projects.
- Recent folder run management.
- Full parity with every project export.
- Batch folder runs.
- Advanced ecological insights inside folder runs.

## Final recommendation

Build a task-based home screen and make **Analyse a folder** the default path.

Do not call it **Simple mode**. Do not call the current app **Advanced mode**.

Use:

- **Analyse a folder** for the legacy-style workflow.
- **Research projects** for the current project/deployment workflow.
- **Timelapse integration** for Timelapse Analyser.

This keeps the old AddaxAI simplicity for the majority workflow while preserving the richer project system for users who actually need it. It also gives the product a clearer promise: AddaxAI can be a quick AI tool when that is enough, and a research workspace when the study needs structure.
