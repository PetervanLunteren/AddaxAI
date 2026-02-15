# Verification plan for camera trap AI predictions

## Context and goals

This document summarises the design decisions and recommendations for a human-in-the-loop verification workflow in a camera trap analysis platform that uses:
- object detection to find animals
- crop-level classification to assign species
- instance-level depth estimation
- images and short video clips from camera traps

Key constraints and goals:
- instance-level structure is required for depth estimation
- no need to generate ML training data from user edits
- optimise for user experience and throughput
- avoid confusing users with multiple annotation modes
- visualisations are secondary to usability and correctness

The primary user need is ecological truth: which species are present in an image or sequence, and how many individuals there are.

---

## Core design principle

There is **one single verification workflow**.

The **data outcome is instance-level**, but the **user experience is confirmation-first**, not annotation-first.

Users confirm or correct **model-proposed candidates**, rather than manually creating or perfecting annotations.

Bounding boxes are treated as system artifacts required for inference, not as user-created ground truth.

---

## Recommended verification depth

### Instance-aware, confirmation-first workflow

For each image or sequence, the system proposes a set of **animal candidates** (internally instances).

For each candidate, the user can:
- confirm it is a real animal
- delete it as a false positive
- correct the species label if needed

At the image or sequence level, the user also confirms:
- which species are present
- the final number of individuals per species
- blank vs non-blank status

This ensures instance-level correctness while keeping the user task aligned with ecological reasoning.

---

## Adding missing individuals

If the user indicates that more individuals are present than proposed by the model, missing instances are added using low-effort interactions.

Supported approaches:
- **tap-to-add**: user clicks once on the animal, the system creates a default bounding box
- **auto-proposal**: the system suggests additional low-confidence candidates that the user can confirm

Manual box drawing is never the default interaction.

---

## Role of bounding boxes

Bounding boxes exist because they are required for:
- depth estimation
- instance separation
- internal consistency

However:
- boxes are always system-generated
- geometry is treated as “good enough” by default
- users are not required to perfect box placement

Optional:
- allow light box adjustment only when depth quality is flagged as poor

Boxes are never framed as user-verified ground-truth annotations.

---

## Video handling strategy

### Core idea

For videos, **measurement and verification are image-based**, not frame-by-frame.

Each video is automatically reduced to one or a small number of **canonical frames** that are used for:
- detection
- instance verification
- depth estimation

The video itself is retained as supporting evidence and context.

---

### Canonical frame selection

The system automatically selects the best frame based on:
- highest detector confidence
- minimal motion blur
- maximal animal visibility
- largest projected animal size
- minimal occlusion

This frame is treated as the authoritative frame for instance structure and depth.

---

### User experience for video

From the user’s perspective:
- the video appears as a single representative image for verification
- the full clip can be scrubbed or played if needed for context
- counts and species are confirmed based on the canonical frame

Users are never asked to annotate bounding boxes across multiple frames.

---

### Depth estimation for video

Depth estimation is performed:
- on the canonical frame only
- per verified instance in that frame

Depth values may be smoothed or validated using neighbouring frames internally, but this is not exposed in the UI.

If depth confidence is low, the system:
- flags the estimate as uncertain
- allows depth to be marked as unknown

---

### Handling multi-individual videos

If all individuals are visible in the canonical frame:
- standard image workflow applies

If additional individuals appear only in other frames:
- the user can indicate that an extra individual exists
- the system selects a secondary canonical frame for that individual
- the instance is verified using that frame only

This avoids per-frame annotation while preserving correctness.

---

## Result representation after verification

Final outputs are expressed as **verified observations with instances**.

Example:
- media id (image or video)
- species: bear
- individuals: 3
- instances: 3 (user-confirmed)
- depth estimates per instance
- verification status

Bounding boxes may be shown as contextual evidence but are not the primary result.

---

## Data model separation

Maintain two conceptual layers:

### Model layer
- proposed instances
- bounding boxes
- species predictions and confidences
- depth estimates and quality signals
- canonical frame references for videos

### User-confirmed layer
- accepted instances
- corrected species labels
- added or removed instances
- final counts per species
- notes or uncertainty flags

Downstream reporting and exports use the user-confirmed layer.

---

## Why frame-level video annotation is avoided

Frame-by-frame bounding box annotation:
- is extremely time-consuming
- provides little additional value for ecological outputs
- significantly harms user experience
- introduces noisy and inconsistent geometry

Reducing video to canonical frames achieves most of the value with far lower complexity.

---

## Final recommendation

Implement a **single, confirmation-first, instance-aware workflow** that:
- treats images and videos uniformly
- reduces videos to canonical frames for verification and depth
- minimises required user actions
- treats bounding boxes as system artifacts
- produces instance-level results suitable for depth estimation and ecological reporting

This approach maximises usability while meeting technical requirements.
