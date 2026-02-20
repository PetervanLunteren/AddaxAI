# Offline Camera Trap Management Platform  
## UX/UI Design Decisions, Glossary, and UI Handoff

This document summarizes the UX/UI design decisions for an offline camera trap management platform and provides a glossary, naming conventions, and a UI design handoff for designers.

---

## 1. Core Design Goals

- clarity over cleverness  
- minimal cognitive load  
- fast happy-path workflow  
- scalable to large datasets and taxonomies  
- professional, tool-like behavior inspired by GIS and ML pipeline tools  

---

## 2. Global Navigation

Primary navigation is minimal and persistent:

- dashboard  
- new analysis  
- images  
- settings  

No secondary or hidden navigation.

---

## 3. New Analysis: High-Level Pattern

The new analysis screen follows a **two-pane batch queue pattern**:

- left pane: what will run (queue context)  
- right pane: how the selected item is configured (focus context)  
- bottom bar: execution control  

This mirrors professional render queues and GIS batch processors.

---

## 4. Queue Panel (Left Pane)

### Purpose
Display batch context, execution order, and processing state.

### Widgets
- add deployment button  
- queue list  
- queue item cards  

### Queue Item Card
- deployment name (derived from folder)  
- status icon (ready, missing input, running, completed, error)  
- inline progress bar when running  
- overflow menu: edit, duplicate, remove  

No configuration controls appear here.

---

## 5. Deployment Setup Panel (Right Pane)

Edits the currently selected queue item.

### Data Source
- folder picker  
- read-only path display  
- automatic image count summary  

### Site Selection
- site dropdown  
- add new site button  

Add new site opens a modal:
- site name field  
- interactive map with click-to-place marker  
- existing sites shown  
- latitude and longitude auto-filled  

### Models
Models are **not selected per deployment**.

- models are defined at project level  
- deployment view shows read-only model summary  

---

## 6. Execution Controls

Persistent bottom bar:

- run queue button (disabled until valid)  
- during execution:
  - global progress bar  
  - current deployment name  
  - pause and cancel actions  

Each queue item shows individual progress inline.

---

## 7. Taxonomy and Class Selection

### Constraints
- fixed 5-level hierarchy  
- dozens to hundreds of classes  
- class availability depends on model  
- usually configured once per project  

### Key Decision
Taxonomy selection is **project-scoped**, not deployment-scoped.

---

## 8. Taxonomy Editor Location

Primary:
- settings → project → taxonomy  

Secondary:
- read-only summary in new analysis  
- optional per-deployment override (advanced only)  

---

## 9. Taxonomy Editor Design

A **full-screen modal or dedicated page** is used due to depth and importance.

### Structure
- hierarchical tree with fixed 5 levels  
- tri-state parent checkboxes  
- binary leaf nodes  

### Behavior
- parent toggles all descendants  
- indeterminate states visible  
- collapsed branches preserve state  

### Search
- filters taxonomy  
- auto-expands ancestor chain  
- highlights matches  
- hides non-matching branches  

### Bulk Actions
Always visible:
- include all  
- exclude all  
- invert  

### Guard Rails
- warn if zero classes selected  
- disable unsupported classes  
- show compatibility warnings without blocking  

---

## 10. Models and Taxonomy Relationship

### Strong Default
- one detection model per project  
- one classification model per project  
- one taxonomy per project  

Model choice defines available classes.

### Rationale
- avoids ambiguity  
- simplifies validation  
- ensures comparability  
- aligns with user mental models  

---

## 11. Project Settings Structure

```
project settings
├─ general
├─ model
│  ├─ detection model
│  ├─ classification model
│  └─ installation status
├─ taxonomy
│  └─ class selection
```

Recommended setup flow:
1. select models  
2. install models if needed  
3. configure taxonomy  
4. add deployments  

---

## 12. Advanced Use Cases

### Advanced Mode
Optional toggle in project settings:
- allow per-deployment models  
- warnings shown  
- increased UI complexity intentional  

### Preferred Alternative
- duplicate project with a different model setup  

---

## 13. Glossary

**Project**  
A logical container defining models, taxonomy, sites, and deployments.

**Deployment**  
A single camera trap dataset defined by an image folder and site.

**Queue**  
An ordered list of deployments to be processed sequentially.

**Detection Model**  
Model responsible for detecting objects or animals in images.

**Classification Model**  
Model responsible for assigning detected objects to classes.

**Taxonomy**  
Hierarchical structure of classes supported by a model.

**Class**  
A leaf node in the taxonomy representing a detectable or classifiable entity.

**Tri-state Checkbox**  
Checkbox with three states: checked, unchecked, indeterminate.

**Project Scope**  
Configuration shared by all deployments within a project.

**Deployment Override**  
Optional configuration differing from project defaults, applied to one deployment.

---

## 14. Naming Conventions

### UI Labels
- use sentence case  
- avoid abbreviations  
- prefer nouns for sections and verbs for actions  

Examples:
- add deployment  
- run queue  
- edit taxonomy  
- project settings  

### Status Labels
- ready  
- missing input  
- installing  
- running  
- completed  
- error  

### Internal Concepts (for designers and developers)
- queue item, not job  
- taxonomy, not label list  
- deployment, not dataset  

---

## 15. UI Design Handoff

### Target Audience
Professional users working with offline data, scientific workflows, and long-running analyses.

### Visual Tone
- calm  
- neutral  
- information-dense but uncluttered  

### Layout Rules
- left = context  
- right = focus  
- bottom = execution  

### Interaction Principles
- inline feedback over dialogs  
- progressive disclosure  
- no wizard-style flows  
- modals only for high-impact configuration  

### Components to Design

- queue list and item cards  
- hierarchical taxonomy tree (5 levels)  
- tri-state checkbox behavior  
- full-screen taxonomy editor  
- site picker map modal  
- progress bars (item-level and global)  

### States to Design Explicitly

- empty project  
- empty queue  
- invalid configuration  
- model installing  
- queue running  
- partial completion with errors  

### Accessibility Notes
- keyboard navigable trees  
- visible focus states  
- color not sole indicator of status  

---

## 16. Final Summary

- two-pane queue-based analysis UI  
- project-level models and taxonomy  
- full-screen taxonomy editor  
- minimal, fast deployment workflow  
- strong defaults with explicit advanced escapes  

This document is intended to be directly usable as a UX specification and UI design handoff.
