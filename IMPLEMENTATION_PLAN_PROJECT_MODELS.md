# Implementation Plan: Project-Level Models & Taxonomy Configuration

**Date:** 2024-12-24
**Status:** IN PROGRESS
**Goal:** Add model selection and taxonomy configuration to project creation, making models project-scoped instead of deployment-scoped.

---

## Overview

Refactor project creation to align with UX/UI design document (`camera_trap_ux_ui_design.md`):
- Models are **project-scoped** (not deployment-scoped)
- Detection model + classification model selected during project creation
- Taxonomy configuration (species class selection) required during project setup
- UI: Accordion-based dialog (not wizard)

---

## Key Design Decisions

### 1. Migration Defaults
- ✅ Detection model: `MD5A-0-0` (MegaDetector 5a)
- ✅ Classification model: `None` (nullable, "No classification" option available)

### 2. Taxonomy CSV Structure
From HuggingFace: https://huggingface.co/Addax-Data-Science/NAM-ADS-v1/raw/main/taxonomy.csv

**Format:**
- 6 columns: `model_class`, `class`, `order`, `family`, `genus`, `species`
- `model_class` = common name (user-facing, e.g., "leopard")
- Remaining 5 = scientific taxonomy hierarchy
- Some rows have empty `genus` or `species`

### 3. Taxonomy Storage Format
- ✅ **Flat list** of selected `model_class` values
- Stored in `project.taxonomy_config` as:
  ```json
  {
    "selected_classes": ["leopard", "elephant", "baboon"]
  }
  ```

### 4. UI Design
- Accordion-based single dialog (4 sections: General, Detection Model, Classification Model, Taxonomy)
- Model dropdowns show: `{emoji} {friendly_name}` + caption with description (~80 chars)
- Status badges: "Ready", "Download required", "Install required"
- Inline action buttons: "Download Weights", "Build Environment"
- Full-screen taxonomy editor modal (tree component - deferred to later)

---

## Implementation Checklist

### ✅ Backend Database Schema

- [x] Migration: Add `detection_model_id`, `classification_model_id`, `taxonomy_config` to `projects` table
  - File: `backend/alembic/versions/20251224_1134_cc3dc3ddf432_add_models_to_projects.py`
  - Default: `detection_model_id='MD5A-0-0'`, `classification_model_id=NULL`, `taxonomy_config='{}'`

- [x] Migration: Remove `detection_model_id`, `classification_model_id`, `species_list` from `deployment_queue`
  - File: `backend/alembic/versions/20251224_1135_3ee4969ac7fb_remove_models_from_deployment_queue.py`

- [x] Update `Project` model with new fields
  - File: `backend/app/models/project.py`
  - Added: `detection_model_id: Mapped[str]`, `classification_model_id: Mapped[str | None]`, `taxonomy_config: Mapped[dict]`

- [x] Update `DeploymentQueue` model (remove model fields)
  - File: `backend/app/models/deployment_queue.py`
  - Removed: `detection_model_id`, `classification_model_id`, `species_list`

### ✅ Backend Taxonomy Parser

- [x] Create taxonomy CSV parser
  - File: `backend/app/ml/taxonomy_parser.py`
  - Functions: `parse_taxonomy_csv()`, `get_all_leaf_classes()`
  - Returns hierarchical tree structure with 6 levels

### ✅ Backend API Endpoints

- [x] Update Pydantic schemas for Project API
  - File: `backend/app/api/schemas/project.py`
  - Update `ProjectCreate`, `ProjectUpdate`, `ProjectResponse` with model fields

- [x] Add list detection models endpoint
  - File: `backend/app/api/routers/ml_models.py`
  - `GET /api/ml/models/detection` → returns `list[ModelInfo]`

- [x] Add list classification models endpoint
  - File: `backend/app/api/routers/ml_models.py`
  - `GET /api/ml/models/classification` → returns `list[ModelInfo]` (includes "None" option)

- [ ] Add taxonomy endpoint
  - File: `backend/app/api/routers/ml_models.py`
  - `GET /api/ml/models/{model_id}/taxonomy` → returns tree structure + all_classes list
  - Reads taxonomy.csv from `~/AddaxAI/models/cls/{model_id}/taxonomy.csv`

- [ ] Update project CRUD with model validation
  - File: `backend/app/api/routers/projects.py`
  - Validate model IDs exist before creating project
  - Normalize "none" to NULL for classification_model_id

### ✅ Frontend Types & API Clients

- [x] Update TypeScript types
  - File: `frontend/src/api/types.ts`
  - Update `ProjectCreate`, `ProjectResponse` interfaces
  - Add `ModelInfo`, `TaxonomyNode`, `TaxonomyResponse` interfaces

- [x] Create models API client
  - File: `frontend/src/api/models.ts`
  - Methods: `listDetectionModels()`, `listClassificationModels()`, `getTaxonomy()`, `getModelStatus()`, `prepareWeights()`, `prepareEnvironment()`

### 🔄 Frontend UI Components

- [x] Refactor `CreateProjectDialog` (minimal version)
  - File: `frontend/src/components/projects/CreateProjectDialog.tsx`
  - Added basic model selection dropdowns (detection + classification)
  - Shows emoji + friendly name
  - Form validation with zod
  - **TODO: Add accordion layout, status badges, taxonomy editor**

- [ ] Create `ModelSelect` component (enhanced version)
  - File: `frontend/src/components/projects/ModelSelect.tsx`
  - Dropdown with emoji + friendly name + description caption
  - Show status badges (ready/download/install)
  - Inline action buttons for preparation

- [ ] Create basic `TaxonomyEditor` component
  - File: `frontend/src/components/taxonomy/TaxonomyEditor.tsx`
  - Full-screen modal
  - Search + bulk actions (include all, exclude all, invert)
  - **For MVP: Simple flat list with checkboxes** (tree component deferred)
  - Shows count of selected classes

- [ ] Update `EditProjectDialog`
  - File: `frontend/src/components/projects/EditProjectDialog.tsx`
  - Show read-only model info
  - Add "Change models" and "Edit taxonomy" buttons

### 🔲 Remove Model Selection from Deployment Flow

- [ ] Update `AnalysesPage`
  - File: `frontend/src/pages/AnalysesPage.tsx`
  - Add read-only project models display at top
  - Remove model selection from deployment wizard

- [ ] Update `DeploymentWizard`
  - File: `frontend/src/components/analyses/DeploymentWizard.tsx`
  - Remove model selection steps
  - Keep only: Folder selection + Site/Deployment

- [ ] Update deployment queue API types
  - File: `frontend/src/api/deployment-queue.ts`
  - Remove `detection_model_id`, `classification_model_id`, `species_list` from payload

### 🔲 Backend Queue Processing

- [ ] Update deployment queue processing worker
  - Fetch models from `project.detection_model_id` and `project.classification_model_id`
  - Use `project.taxonomy_config` for species filtering
  - Fail gracefully if project models not configured

### 🔲 Testing

- [ ] Create project with models end-to-end
- [ ] Configure taxonomy and verify storage
- [ ] Create deployment (models inherited from project)
- [ ] Run deployment queue with project-level models
- [ ] Test model preparation (download + install)

---

## Deferred Tasks (Post-MVP)

### Hierarchical Taxonomy Tree Component
**Complexity:** HIGH
**File:** `frontend/src/components/taxonomy/TaxonomyTree.tsx`

Requirements:
- Recursive rendering (6 levels deep)
- Tri-state checkboxes (checked, unchecked, indeterminate)
- Parent toggles all descendants
- Search with auto-expand + highlight
- Collapse/expand branches
- Performance optimization for hundreds of nodes

**Alternative:** Use library like `react-complex-tree` or `@atlaskit/tree`

**For MVP:** Use simple flat list with checkboxes showing all `model_class` values

---

## Current Status (2024-12-24 12:00)

### ✅ Completed (Minimal Implementation)
1. Backend taxonomy parser
2. Database migrations (projects + deployment_queue)
3. Project and DeploymentQueue model updates
4. Backend Pydantic schemas
5. Backend API endpoints (detection + classification models)
6. Frontend TypeScript types
7. Frontend models API client
8. CreateProjectDialog basic refactor (dropdowns working)
9. Fixed manifest validation (relaxed type checking)
10. Simplified ModelManifest schema (removed confidence_threshold, minimal validation)

### 📋 Next Tasks
- [ ] Add taxonomy endpoint (backend)
- [ ] Update project CRUD with model validation (backend)
- [ ] Create TaxonomyEditor component (frontend)
- [ ] Enhance CreateProjectDialog with accordion + taxonomy button
- [ ] Remove model selection from deployment flow
- [ ] Update queue processing worker
- [ ] End-to-end testing

**Total:** 7 major tasks remaining

---

## Notes

- SQLite limitation: Can't remove server_default after column creation (kept in schema)
- Model manifests stored in `~/AddaxAI/models/det/` and `~/AddaxAI/models/cls/`
- Taxonomy CSV downloaded from HuggingFace during model preparation
- Existing projects migrated with default `MD5A-0-0` detection model, no classification
- Breaking change: Existing deployment_queue entries lose model configuration

---

## References

- UX/UI Design: `camera_trap_ux_ui_design.md`
- Developer Guidelines: `DEVELOPERS.md`
- Project Plan: `PROJECT_PLAN.md`
- Taxonomy Example: https://huggingface.co/Addax-Data-Science/NAM-ADS-v1/raw/main/taxonomy.csv
