 Browse and verify -- implementation plan                                                                                                              

 Context

 The current ReviewPage (frontend/src/pages/ReviewPage.tsx) is a placeholder showing observation type stats and a "coming soon" message. The ImagesPage
 shows a flat gallery of files with no event grouping. There is no way for users to verify ML predictions, edit bounding boxes, or annotate detections.

 This plan adds a "Browse and verify" page that replaces /review with an event-centric browse and annotation workflow. Users can browse events
 (time-clustered file groups) as cards, open an event detail modal with a filmstrip, and toggle into verify mode for interactive bounding box editing
 with react-konva.

 The Event model and tables already exist in the database (created by the initial migration) but are never populated. This plan implements the event
 grouping algorithm and all supporting infrastructure.

 Delivery: 3 phased milestones, each independently testable.

 ---
 Phase 1: Event grouping backend + browse page with event cards

 1.1 Database migration

 New file: backend/alembic/versions/20260215_1200_<rev>_add_file_verification_fields.py

 Add to files table:
 - verified (Boolean, NOT NULL, default False) + index idx_files_verified
 - verified_at (DateTime, nullable)
 - notes (Text, nullable)

 Chain from current head: down_revision = "b2c3d4e5f6a7"

 1.2 File model changes

 Modify: backend/app/models/file.py

 Add three fields to the File class:
 verified: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0", default=False)
 verified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
 notes: Mapped[str | None] = mapped_column(Text, nullable=True)
 Add Index("idx_files_verified", "verified") to __table_args__.

 1.3 File schemas + PATCH endpoint

 Modify: backend/app/api/schemas/file.py
 - Add verified: bool, verified_at: datetime | None, notes: str | None to FileResponse
 - Add new FileUpdate schema: verified: bool | None = None, notes: str | None = None

 Modify: backend/app/api/crud/file.py
 - Add update_file(db, file_id, update) -- sets verified_at = utcnow() when verified changes to True, clears when False

 Modify: backend/app/api/routers/files.py
 - Add PATCH /api/files/{file_id} using FileUpdate schema, returns FileResponse

 1.4 Event grouping algorithm

 New file: backend/app/api/crud/event.py

 Core function generate_events_for_project(db, project_id):
 1. Fetch project to get independence_interval (seconds)
 2. Delete all existing events for every deployment in the project (idempotent regeneration)
 3. For each deployment, query files ordered by timestamp ASC
 4. Walk files: start new event when gap > independence_interval seconds
 5. Create Event records with start_time, end_time, file_count
 6. Insert event_files junction rows with sequence_number (0-indexed)
 7. Return event count

 Additional CRUD functions:
 - get_events_by_project(db, project_id, skip, limit) -- join through Deployment.site.has(project_id=...) (reuse existing pattern from
 backend/app/api/crud/file.py:61-89)
 - get_event_with_files(db, event_id) -- event with all files + their detections, files ordered by sequence_number
 - get_event_count_by_project(db, project_id) -- count for pagination

 Representative file logic: For each event summary, pick the file with the highest sum of animal detection confidences. If no detections, use first file
  (chronologically). For videos, the API already serves best_frame_path via GET /api/files/{id}/image.

 1.5 Event schemas

 New file: backend/app/api/schemas/event.py

 EventSummary:
   id, deployment_id, start_time, end_time, file_count
   representative_file_id: str      # file with best detections
   species: list[str]               # unique species across event
   observation_type: str             # dominant type (animal > human > vehicle > blank)
   verified_count: int               # files with verified=True
   total_count: int

 EventWithFiles:
   id, deployment_id, start_time, end_time, file_count, created_at
   files: list[FileWithDetections]   # reuse existing schema from schemas/file.py

 1.6 Event API router

 New file: backend/app/api/routers/events.py

 ┌────────┬────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
 │ Method │        Endpoint        │                                      Purpose                                       │
 ├────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ POST   │ /api/events/generate   │ Generate/regenerate events for a project. Body: {project_id}. Idempotent.          │
 ├────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ GET    │ /api/events            │ List event summaries. Params: project_id, skip, limit. Returns list[EventSummary]. │
 ├────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ GET    │ /api/events/count      │ Total event count. Params: project_id. Returns {count: int}.                       │
 ├────────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
 │ GET    │ /api/events/{event_id} │ Get event with all files + detections. Returns EventWithFiles.                     │
 └────────┴────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

 Register in backend/app/api/routers/__init__.py and backend/app/main.py.

 1.7 Frontend types

 Modify: frontend/src/api/types.ts
 - Add EventSummary and EventWithFiles interfaces
 - Add verified: boolean, verified_at: string | null, notes: string | null to FileResponse

 1.8 Frontend API client

 New file: frontend/src/api/events.ts
 - eventsApi.generate(projectId)
 - eventsApi.list({ project_id, skip?, limit? })
 - eventsApi.count(projectId)
 - eventsApi.get(eventId)

 Modify: frontend/src/api/files.ts
 - Add filesApi.update(id, { verified?, notes? })

 1.9 "Browse and verify" page

 New file: frontend/src/pages/VerifyPage.tsx (replaces ReviewPage.tsx)

 Layout:
 +--------------------------------------------------+
 | Browse and verify                                 |
 | [Generate events] (shown when 0 events exist)    |
 +--------------------------------------------------+
 | Event card grid (responsive 2-4 columns)          |
 |                                                    |
 | +------------+ +------------+ +------------+      |
 | | [thumbnail]| | [thumbnail]| | [thumbnail]|      |
 | | 7 files    | | 3 files    | | 12 files   |      |
 | | 15:32-15:34| | 16:01      | | 16:45-16:52|      |
 | | Zebra, Lion| | Elephant   | | Blank      |      |
 | | ▓▓▓░░ 3/7  | | ▓▓▓ 3/3   | | ░░░░ 0/12  |      |
 | +------------+ +------------+ +------------+      |
 |                                                    |
 | Load more pagination                               |
 +--------------------------------------------------+

 Each EventCard shows:
 - Thumbnail via <img src="/api/files/{representative_file_id}/image" />
 - File count ("7 files")
 - Time range (start_time -- end_time, or single time if same)
 - Species tags (max 3 shown, "+ N more" if more)
 - Verification progress bar (verified_count / total_count)
 - Observation type badge (reuse color scheme from ImagesPage)

 Click card sets selectedEventId (phase 2 opens modal).

 1.10 Routing and navigation changes

 Modify: frontend/src/App.tsx
 - Replace import ReviewPage with import VerifyPage from "./pages/VerifyPage"
 - Change route: <Route path="verify" element={<VerifyPage />} />
 - Add redirect: <Route path="review" element={<Navigate to="../verify" replace />} />

 Modify: frontend/src/components/layout/Sidebar.tsx
 - Change nav item: { to: ..../verify, icon: CheckCircle, label: "Browse and verify" }

 Delete: frontend/src/pages/ReviewPage.tsx

 1.11 Shared detection utilities

 New file: frontend/src/lib/detection-utils.ts

 Extract from ImagesPage.tsx (lines 16-59):
 - getCategoryColor(category: string): string
 - getDetectionLabel(detection: DetectionResponse): string

 Update ImagesPage.tsx to import from this shared file.

 Phase 1 verification

 1. cd backend && source venv/bin/activate && PYTHONPATH=. alembic upgrade head -- migration applies
 2. Start backend: uvicorn app.main:app --reload
 3. curl -X POST http://localhost:8000/api/events/generate -d '{"project_id":"<id>"}' -- events generated
 4. curl http://localhost:8000/api/events?project_id=<id> -- returns event summaries
 5. Start frontend: cd frontend && npm run dev
 6. Navigate to /projects/<id>/verify -- see event cards with thumbnails
 7. Old /review URL redirects to /verify
 8. Sidebar shows "Browse and verify" as active

 ---
 Phase 2: Event detail modal with filmstrip and queue navigation

 2.1 Navigation endpoint

 Modify: backend/app/api/routers/events.py

 Method: GET
 Endpoint: /api/events/{event_id}/adjacent
 Purpose: Get adjacent event IDs. Params: project_id. Returns {previous_id, next_id, next_unverified_id, current_index, total_count}.

 Modify: backend/app/api/crud/event.py
 - Add get_adjacent_events(db, event_id, project_id) -- queries all events ordered by start_time DESC, finds neighbors. "Next unverified" = next event
 where any file has verified=False.

 2.2 Frontend API additions

 Modify: frontend/src/api/events.ts
 - Add eventsApi.getAdjacent(eventId, projectId)

 2.3 Zustand store

 New file: frontend/src/stores/verify-store.ts

 interface VerifyStore {
   selectedEventId: string | null;
   selectedFileIndex: number;
   isDetailOpen: boolean;
   openDetail: (eventId: string) => void;
   closeDetail: () => void;
   setSelectedFileIndex: (index: number) => void;
 }

 2.4 Event detail modal

 New file: frontend/src/components/verify/EventDetailModal.tsx

 Near-full-screen Dialog (max-w-[95vw] max-h-[95vh]). Layout inspired by the Agouti screenshot:

 +------------------------------------------------------------+
 | [X]   Event 4 of 87   < Prev | Next > | >> Next unverified |
 +------------------------------------------------------------+
 |                                                             |
 |  +------------------------------------------------------+  |
 |  |                                                        |  |
 |  |       Main image with SVG detection overlay            |  |
 |  |       (read-only bounding boxes, same approach         |  |
 |  |        as ImagesPage but using shared utils)           |  |
 |  |                                                        |  |
 |  +------------------------------------------------------+  |
 |                                                             |
 |  Filmstrip: [img1] [img2*] [img3] [img4] ... [img7]       |
 |             Image 2 of 7                                    |
 |                                                             |
 |  Species: Zebra (4), Lion (1) | Verified: 3 of 7 files    |
 +------------------------------------------------------------+

 2.5 Component breakdown

 New file: frontend/src/components/verify/EventNavigationBar.tsx
 - "Event N of M" counter
 - Previous / Next / Next unverified buttons
 - Keyboard: ArrowLeft = prev event, ArrowRight = next event, N = next unverified

 New file: frontend/src/components/verify/EventFilmstrip.tsx
 - Horizontal scrollable row of file thumbnails (96x64px)
 - Selected file highlighted with blue ring
 - Verified files show green checkmark overlay
 - "Image N of M" label below
 - Keyboard: ArrowUp = prev file, ArrowDown = next file

 New file: frontend/src/components/verify/EventImageViewer.tsx
 - Image display with SVG detection overlay (read-only)
 - Uses shared getCategoryColor and getDetectionLabel from lib/detection-utils.ts
 - Filters detections by project's detection_threshold

 2.6 Wiring

 Modify: frontend/src/pages/VerifyPage.tsx
 - On event card click: verifyStore.openDetail(eventId)
 - Render EventDetailModal with isDetailOpen from store

 Phase 2 verification

 1. Click an event card -- modal opens with representative image + detection boxes
 2. Click filmstrip thumbnails -- main image updates
 3. Press arrow keys -- navigate between files and events
 4. "Event N of M" counter is accurate
 5. "Next unverified" skips to the correct event
 6. Close modal returns to card grid
 7. Test with single-file events and multi-file events

 ---
 Phase 3: Verify mode with interactive annotation

 3.1 Install react-konva

 cd frontend && npm install react-konva konva

 3.2 Make detection job_id nullable

 New migration: backend/alembic/versions/20260215_1400_<rev>_make_detection_job_id_nullable.py

 Use batch_alter_table for SQLite compatibility:
 with op.batch_alter_table("detections") as batch_op:
     batch_op.alter_column("job_id", nullable=True)

 Modify: backend/app/models/detection.py
 - Change job_id to Mapped[str | None] with nullable=True

 3.3 Detection CRUD API

 New file: backend/app/api/routers/detections.py

 Method: POST
 Endpoint: /api/detections
 Purpose: Create human-drawn detection. Body: {file_id, category, bbox_x/y/w/h, species?}. Sets classification_method="human", job_id=None.
 ────────────────────────────────────────
 Method: PATCH
 Endpoint: /api/detections/{id}
 Purpose: Update detection. Body: any subset of {category, bbox_x/y/w/h, species, species_confidence}. Sets classification_method="human" when species
   edited.
 ────────────────────────────────────────
 Method: DELETE
 Endpoint: /api/detections/{id}
 Purpose: Delete detection. Returns 204.

 Modify: backend/app/api/schemas/detection.py
 - Add DetectionCreateHuman schema (like DetectionCreate but no job_id)
 - Add DetectionUpdate schema (all fields optional)

 Modify: backend/app/api/crud/detection.py
 - Add create_human_detection(db, data) -- creates with job_id=None, classification_method="human", confidence=1.0
 - Add update_detection(db, detection_id, update) -- partial update

 Modify: backend/app/api/crud/file.py
 - Add recalculate_observation_type(db, file_id) -- re-derive from current detections (animal > human > vehicle > blank). Call after detection
 create/update/delete.

 Register in backend/app/api/routers/__init__.py and backend/app/main.py.

 3.4 Frontend detection API client

 New file: frontend/src/api/detections.ts
 - detectionsApi.create({ file_id, category, bbox_x/y/w/h, species? })
 - detectionsApi.update(id, { category?, bbox_x/y/w/h?, species? })
 - detectionsApi.delete(id)

 Modify: frontend/src/api/types.ts
 - Add DetectionCreate, DetectionUpdate interfaces

 3.5 Annotation canvas

 New file: frontend/src/components/verify/AnnotationCanvas.tsx

 react-konva <Stage> replacing the SVG overlay when verify mode is active:
 - <Layer> for background image (loaded via window.Image constructor)
 - <Layer> for bounding boxes as Konva <Rect> elements
   - draggable={true} for moving boxes
   - <Transformer> on selected box for resize handles
   - Category-based colors matching existing scheme
   - Labels with species + confidence
 - Drawing mode: mousedown on empty area starts new box, mouseup creates via API
 - Coordinate conversion: normalized = pixel / imageDimension (and reverse)
 - Delete/Backspace on selected box calls delete API

 3.6 Verification panel

 New file: frontend/src/components/verify/FileVerificationPanel.tsx

 Right-side panel (visible in verify mode):
 - Detection list for current file (click to select on canvas)
 - Species input per detection (autocomplete using known species from projectsApi.getSpeciesStats)
 - Category badge per detection
 - Delete button per detection
 - "Draw new box" button (activates drawing mode)
 - Notes textarea (saved via filesApi.update)
 - "Mark verified" / "Mark unverified" button (saved via filesApi.update)
 - Keyboard: V = toggle verified, D = draw mode, Delete = delete selected

 3.7 Verify mode toggle

 Modify: frontend/src/components/verify/EventDetailModal.tsx

 Add verify mode toggle (Switch component from shadcn/ui):
 - View mode (default): SVG overlay (read-only), no side panel
 - Verify mode: Konva canvas (interactive), FileVerificationPanel on right

 Layout in verify mode:
 +------------------------------------------------------------+
 | Event 4 of 87   [View | Verify]  < Prev | Next > | >> Next |
 +------------------------------------------------------------+
 |                                           |                 |
 |  Image with Konva annotation canvas       | Detections:     |
 |  (draw, resize, move, delete boxes)       |  [species edit] |
 |                                           |  [delete]       |
 |                                           |                 |
 |  Filmstrip: [img1] [img2*] ...            | Notes: [___]    |
 |  Image 2 of 7                             | [Mark verified] |
 +------------------------------------------------------------+

 3.8 Keyboard shortcuts (all phases)

 ┌──────────────────┬───────────────────────────┬─────────────────────────────────────────────┐
 │       Key        │          Context          │                   Action                    │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ ArrowLeft        │ Modal open                │ Previous event                              │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ ArrowRight       │ Modal open                │ Next event                                  │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ ArrowUp          │ Modal open                │ Previous file in filmstrip                  │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ ArrowDown        │ Modal open                │ Next file in filmstrip                      │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ N                │ Modal open                │ Next unverified event                       │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ V                │ Verify mode               │ Toggle verified on current file             │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ D                │ Verify mode               │ Enter draw mode                             │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ Delete/Backspace │ Verify mode, box selected │ Delete selected detection                   │
 ├──────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
 │ Escape           │ Modal/verify              │ Deselect box / exit draw mode / close modal │
 └──────────────────┴───────────────────────────┴─────────────────────────────────────────────┘

 Shortcuts only fire when modal is open and no text input is focused.

 Phase 3 verification

 1. Apply new migration: PYTHONPATH=. alembic upgrade head
 2. curl -X POST http://localhost:8000/api/detections -d
 '{"file_id":"...","category":"animal","bbox_x":0.1,"bbox_y":0.1,"bbox_width":0.2,"bbox_height":0.2}' -- creates detection
 3. Toggle verify mode in modal -- SVG replaced by Konva canvas with interactive boxes
 4. Draw a new box on the canvas -- detection appears in side panel
 5. Resize/move a box -- coordinates update via API
 6. Delete a box -- removed from canvas and database
 7. Edit species on a detection -- saved as classification_method: "human"
 8. Add notes to a file -- persists after modal close/reopen
 9. Mark file verified -- green checkmark on filmstrip thumbnail
 10. All keyboard shortcuts work correctly

 ---
 Complete file inventory

 New files

 ┌────────────────────────────────────────────────────────────────┬───────┬─────────────────────────────────────────┐
 │                              File                              │ Phase │                 Purpose                 │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/alembic/versions/..._add_file_verification_fields.py   │ 1     │ Migration: verified, verified_at, notes │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/alembic/versions/..._make_detection_job_id_nullable.py │ 3     │ Migration: nullable job_id              │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/app/api/crud/event.py                                  │ 1     │ Event grouping algorithm + CRUD         │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/app/api/schemas/event.py                               │ 1     │ EventSummary, EventWithFiles schemas    │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/app/api/routers/events.py                              │ 1     │ Event API endpoints                     │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ backend/app/api/routers/detections.py                          │ 3     │ Detection CRUD API                      │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/api/events.ts                                     │ 1     │ Events API client                       │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/api/detections.ts                                 │ 3     │ Detections API client                   │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/lib/detection-utils.ts                            │ 1     │ Shared detection color/label utils      │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/pages/VerifyPage.tsx                              │ 1     │ Browse and verify page                  │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/stores/verify-store.ts                            │ 2     │ Zustand store for verify state          │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/EventDetailModal.tsx            │ 2     │ Near-full-screen event modal            │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/EventNavigationBar.tsx          │ 2     │ Prev/Next navigation bar                │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/EventFilmstrip.tsx              │ 2     │ Horizontal file thumbnail strip         │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/EventImageViewer.tsx            │ 2     │ Image with detection overlay            │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/AnnotationCanvas.tsx            │ 3     │ react-konva interactive canvas          │
 ├────────────────────────────────────────────────────────────────┼───────┼─────────────────────────────────────────┤
 │ frontend/src/components/verify/FileVerificationPanel.tsx       │ 3     │ Notes + species edit + verify           │
 └────────────────────────────────────────────────────────────────┴───────┴─────────────────────────────────────────┘

 Modified files

 ┌────────────────────────────────────────────┬───────┬───────────────────────────────────────────────────────┐
 │                    File                    │ Phase │                        Changes                        │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/models/file.py                 │ 1     │ Add verified, verified_at, notes fields               │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/models/detection.py            │ 3     │ Make job_id nullable                                  │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/schemas/file.py            │ 1     │ Add FileUpdate, verification fields to FileResponse   │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/schemas/detection.py       │ 3     │ Add DetectionCreateHuman, DetectionUpdate             │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/crud/file.py               │ 1,3   │ Add update_file(), recalculate_observation_type()     │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/crud/detection.py          │ 3     │ Add create_human_detection(), update_detection()      │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/routers/files.py           │ 1     │ Add PATCH endpoint                                    │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/api/routers/__init__.py        │ 1,3   │ Register events_router, detections_router             │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ backend/app/main.py                        │ 1,3   │ Include events_router, detections_router              │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/src/api/types.ts                  │ 1,3   │ Add event types, verification fields, detection types │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/src/api/files.ts                  │ 1     │ Add update() method                                   │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/src/App.tsx                       │ 1     │ Replace review route with verify                      │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/src/components/layout/Sidebar.tsx │ 1     │ Update nav item                                       │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/src/pages/ImagesPage.tsx          │ 1     │ Import shared detection utils                         │
 ├────────────────────────────────────────────┼───────┼───────────────────────────────────────────────────────┤
 │ frontend/package.json                      │ 3     │ Add react-konva, konva                                │
 └────────────────────────────────────────────┴───────┴───────────────────────────────────────────────────────┘

 Deleted files

 ┌───────────────────────────────────┬───────┬────────────────────────────┐
 │               File                │ Phase │           Reason           │
 ├───────────────────────────────────┼───────┼────────────────────────────┤
 │ frontend/src/pages/ReviewPage.tsx │ 1     │ Replaced by VerifyPage.tsx │
 └───────────────────────────────────┴───────┴────────────────────────────┘