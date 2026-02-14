  Camera trap platform landscape and recommendations for AddaxAI                                                                                         
                                                                                                                                                         
  1. How other platforms organize their menus and navigation

  Platform: Agouti
  Type: Web (cloud)
  Top-level unit: Project
  Main navigation sections: Projects, Deployments, Sequences/Observations, Species, Locations, Export
  ────────────────────────────────────────
  Platform: Camelot
  Type: Desktop (localhost)
  Top-level unit: Survey
  Main navigation sections: Dashboard, Survey, Trap Stations, Sites, Library, Species, Reports, Settings
  ────────────────────────────────────────
  Platform: Wildlife Insights
  Type: Web (cloud)
  Top-level unit: Project
  Main navigation sections: Home/Dashboard, Projects, Initiatives, Explore, Upload, Identify, Download
  ────────────────────────────────────────
  Platform: TrapTagger
  Type: Web (self-hosted)
  Top-level unit: Survey
  Main navigation sections: Dashboard, Surveys, Launch Task, Results, Explore, Export
  ────────────────────────────────────────
  Platform: WildTrax
  Type: Web (hosted)
  Top-level unit: Project
  Main navigation sections: My Projects, Data Discover, Manage, Dashboard (with Camera/ARU/Point Count tabs)
  ────────────────────────────────────────
  Platform: TRAPPER
  Type: Web (self-hosted)
  Top-level unit: Project
  Main navigation sections: Projects, Collections, Resources, Sequences, Classifications, Deployments, Map
  ────────────────────────────────────────
  Platform: eMammal
  Type: Desktop + Web
  Top-level unit: Project
  Main navigation sections: Project Selection, Image Browser, Upload (desktop); Projects, Explore, My Data, Map (web)

  Pattern: Every platform uses either "project" or "survey" as the top-level container. The sidebar or top-level navigation almost always includes: a way
   to manage the organizational hierarchy, an annotation/identification view, a data browsing/exploration view, and an export section.

  2. Terminology consensus

  The Camtrap DP standard (developed by TDWG, closely aligned with Agouti and GBIF) has become the de facto vocabulary reference. Here is the emerging
  consensus:

  ┌──────────────────────────┬────────────────────────────────────┬─────────────────────────────────────────────────────────┬──────────────────────────┐
  │         Concept          │          Camtrap DP term           │                      Also used by                       │  AddaxAI currently uses  │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Top container            │ (datapackage.json)                 │ "Project" everywhere                                    │ Project                  │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Camera location          │ locationID/locationName (field on  │ "Site", "Location", "Station", "Trap Station"           │ Site                     │
  │                          │ deployment)                        │                                                         │                          │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Camera at location for   │ Deployment                         │ Universal across all platforms                          │ Deployment               │
  │ time period              │                                    │                                                         │                          │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Burst of images from one │ Event (via eventID)                │ "Sequence" (Agouti, WI, WildTrax), "Cluster"            │ Event (model exists but  │
  │  trigger                 │                                    │ (TrapTagger)                                            │ not wired up)            │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Individual media file    │ Media                              │ "Image", "Resource", "Photo"                            │ File                     │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Species identification   │ Observation                        │ "Tag" (WildTrax), "Classification" (TRAPPER),           │ Detection                │
  │                          │                                    │ "Identification" (WI), "Sighting" (Camelot)             │                          │
  ├──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┼──────────────────────────┤
  │ Empty/false trigger      │ blank (observationType)            │ "Blank", "Empty", "No animal"                           │ No dedicated concept yet │
  └──────────────────────────┴────────────────────────────────────┴─────────────────────────────────────────────────────────┴──────────────────────────┘

  Key takeaway: Your terminology is already well-aligned. "Site" and "deployment" match the community. "File" for media is fine (Camtrap DP uses "media"
  but "file" is unambiguous). "Detection" currently conflates the MegaDetector bbox output with the species observation — this is worth separating (see
  recommendations below).

  3. Data hierarchy comparison

  Camtrap DP standard hierarchy (the community reference point):
  Project (metadata envelope)
    └── Deployment (camera + location + time period)
          └── Media (individual image/video)
                └── Observation (what was identified)
                      └── Event (groups observations via eventID)

  Your current hierarchy:
  Project
    └── Site (camera location with coordinates)
          └── Deployment (folder of images, linked to a site)
                └── File (individual image/video)
                      └── Detection (bbox + category + species)

  Notable difference: Camtrap DP treats "location" as attributes on deployment, not a separate entity. Most platforms that do have a separate entity call
   it "Location" or "Station." Your use of Site as a separate entity with coordinates is fine and actually matches what most practical platforms do
  (Agouti, WildTrax, TRAPPER, eMammal all have a separate location/site concept). The standard just flattens it.

  4. Core workflow comparison

  Every platform follows essentially the same workflow with minor variations:

  1. Create project (define models, taxonomy, settings)
  2. Define locations/sites (GPS coordinates, habitat)
  3. Create deployments (camera at site for time period, or point to folder)
  4. Import/upload media (scan folders, read EXIF, group into sequences)
  5. Run detection (MegaDetector or equivalent → bboxes)
  6. Run classification (species classifier on detected animals)
  7. Human review (confirm/correct AI predictions)
  8. Analyze (dashboard, statistics, maps)
  9. Export (CSV, Camtrap DP, Darwin Core)

  Where platforms differ most: Step 7 (human review). This is the highest-volume user activity and where UX matters most.

  5. Key UX patterns and what works

  Sequence/event-based annotation (universal best practice)

  Every mature platform annotates at the sequence/event level, not individual images. A burst of 3-10 images from one trigger is treated as a single unit
   of work. This is the single most impactful UX decision.

  - Agouti: "Sequence" is the annotation unit, filmstrip of thumbnails
  - TrapTagger: "Cluster" is the annotation unit, all images shown together
  - WildTrax: "Series" is the annotation unit, thumbnails with series-level tagging
  - Zooniverse/Snapshot Safari: Sequence shown as auto-playing slideshow or grid

  AddaxAI implication: Your Event model exists but isn't wired up. Wiring it up and making event-based browsing the primary view should be a priority.

  Keyboard-driven workflows (efficiency multiplier)

  Timelapse (by Saul Greenberg, University of Calgary) is the gold standard here. Common patterns:

  ┌─────────────────┬─────────────────────────┐
  │       Key       │         Action          │
  ├─────────────────┼─────────────────────────┤
  │ Arrow right / N │ Next sequence           │
  ├─────────────────┼─────────────────────────┤
  │ Arrow left / P  │ Previous sequence       │
  ├─────────────────┼─────────────────────────┤
  │ Enter / Y       │ Accept AI prediction    │
  ├─────────────────┼─────────────────────────┤
  │ E or 0          │ Mark as empty/blank     │
  ├─────────────────┼─────────────────────────┤
  │ 1-9             │ Quick species shortcuts │
  ├─────────────────┼─────────────────────────┤
  │ F               │ Flag for review         │
  ├─────────────────┼─────────────────────────┤
  │ Ctrl+Z          │ Undo                    │
  └─────────────────┴─────────────────────────┘

  The goal: an experienced user should be able to classify an entire dataset without touching the mouse.

  AI-first review with two-threshold system

  The most efficient pattern across platforms (Wildlife Insights, TrapTagger):

  1. Auto-accept threshold (e.g., >95% confidence): AI predictions above this are accepted without human review
  2. Review band (e.g., 50-95%): Human reviews these
  3. Auto-reject threshold (e.g., <50%): Flagged as likely empty or uncertain

  Show a histogram of confidence scores so users can choose thresholds intelligently.

  Confirmation-first, not annotation-first

  Your VERIFICATION_PLAN.md already nails this: users confirm or correct model-proposed candidates, not manually create annotations. This matches the
  pattern at Wildlife Insights, TrapTagger, and Zooniverse.

  Map as navigation, not just visualization

  Clicking a site on the map should filter the data view to that site's data. The map should be an action launcher, not a passive display.

  6. Export format requirements

  ┌─────────────────────┬────────────────────┬────────────────────────────────────────────────────────────┐
  │       Format        │     Importance     │                      Who supports it                       │
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────┤
  │ Camtrap DP          │ Essential          │ Agouti (flagship), TRAPPER, GBIF ingestion pipeline        │
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────┤
  │ CSV                 │ Essential          │ Every platform                                             │
  ├─────────────────────┼────────────────────┼────────────────────────────────────────────────────────────┤
  │ Darwin Core Archive │ Important for GBIF │ Wildlife Insights, eMammal (direct), others via Camtrap DP │
  └─────────────────────┴────────────────────┴────────────────────────────────────────────────────────────┘

  Camtrap DP consists of:
  - datapackage.json (project metadata)
  - deployments.csv (deploymentID, lat/lon, dates, camera info)
  - media.csv (mediaID, deploymentID, timestamp, filePath)
  - observations.csv (observationID, deploymentID, mediaID, eventID, species, count, sex, lifeStage, classificationMethod, classifiedBy,
  classificationProbability)

  Key controlled vocabularies in Camtrap DP:
  - observationType: animal, human, vehicle, blank, unknown, unclassified
  - classificationMethod: human, machine
  - lifeStage: adult, subadult, juvenile, offspring, unknown
  - sex: female, male, unknown

  7. Accessibility considerations

  - Double-stroke bounding boxes: bright inner stroke + dark outer stroke ensures visibility against both light and dark camera trap backgrounds (day,
  night, infrared, snow)
  - Colorblind-safe palettes for species/category colors (avoid red-green; use Okabe-Ito or Wong palettes)
  - All annotation functions keyboard-accessible
  - Label text on bboxes needs solid/semi-transparent background for readability
  - High-contrast toggle for difficult images (night/IR)

  ---
  8. Specific recommendations for AddaxAI

  Based on everything above, here is what aligns well and what could be improved:

  Already well-aligned

  - Project → Site → Deployment hierarchy matches the community
  - Project-scoped models and taxonomy matches best practice (Agouti, WildTrax, Wildlife Insights all do this)
  - Sidebar navigation with project-scoped sections is the right pattern
  - Confirmation-first verification plan matches the best tools
  - Local-first architecture serves a real need (Camelot, Timelapse prove this market)
  - MegaDetector + species classifier pipeline is the industry standard approach

  Terminology refinements to consider

  - Your current "Detection" conflates two things: the bounding box output from MegaDetector (an instance) and the species identification (an
  observation). Consider separating these or at least clearly distinguishing detection-phase outputs from classification-phase outputs in the UI. Camtrap
   DP separates observationType (animal/human/vehicle/blank) from species-level classification.
  - Consider adding "blank" as a first-class concept — marking empty/false-trigger images is the single most common annotation action (50-80% of camera
  trap images are blanks)

  Navigation additions to consider

  - "Identify" or "Review" as a primary sidebar item — this is the highest-volume user activity and deserves top-level navigation rather than being
  buried in the images page
  - The current sidebar has "New analysis, Dashboard, Images, Settings." Consider expanding to: Analyses (queue), Review (annotation/verification),
  Images (browsing/exploration), Dashboard (statistics), Map (spatial), Export, Settings

  Event/sequence support (high priority)

  Wire up the Event model. Make sequence-based browsing the default view for the Review/Identify workflow. Group images by time gap (your 60-second
  default is standard).

  Export (high priority for adoption)

  Camtrap DP export should be a priority — it is the gateway to GBIF publication and interoperability with Agouti, TRAPPER, and the broader community.
  Your data model maps cleanly to it.

  Two-threshold AI review system

  Add confidence threshold controls: auto-accept above X%, require review between X% and Y%, flag below Y%. Show a confidence distribution histogram.

  Keyboard shortcuts for review workflow

  When you build the Review/Identify page, keyboard-driven annotation should be a core requirement, not an afterthought. The patterns are
  well-established (see table above).

  Bounding box rendering

  Switch to double-stroke bounding boxes (bright inner + dark outer) for universal visibility across day/night/IR images. Add a semi-transparent label
  background.

  ---
  Note: All platform details are based on training knowledge up to May 2025. Specific features may have evolved since then, particularly for actively
  developed platforms like Wildlife Insights, Agouti, and TrapTagger. The Camtrap DP standard and GBIF integration patterns are well-established and
  unlikely to have changed significantly.