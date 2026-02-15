# Camera Trap Platform -- Final Verification & Event Design

## Core Principle

Users primarily think and work in terms of **events**. However,
annotation truth must remain at the **file level**.

Events are review containers. Files store ground truth (boxes and
labels).

------------------------------------------------------------------------

# 1. Overall Review Architecture

Use a single grid view with:

-   Filtering (class, confidence, verified status, date, camera, etc.)
-   Click to open Event Review modal
-   Review mode expands to near full-screen
-   Keyboard shortcuts supported
-   Queue snapshot based on active filters

When entering review mode: - Freeze the filtered dataset as a queue
snapshot - Show progress indicator (e.g., 42 / 310 events) - Ensure
stable next/previous navigation

------------------------------------------------------------------------

# 2. Event-Centric Review Model

Events are the primary review unit.

The Event Review screen shows:

Top section: - Large representative image - Editable detection boxes for
that file - Event metadata - Event progress indicator

Below: - Filmstrip or grid of other frames in the event - Small box
overlays - Verification status per frame

------------------------------------------------------------------------

# 3. Verification Rules

You always verify files. You never store verification directly on
events.

### Default Behavior

User verifies the representative image.

System stores: - file_id verified - edited boxes (if any)

Event shows: - Representative verified - 1 of N frames verified

### Optional Deeper Verification

User can click any other frame in the event:

-   Open that frame in the same annotator
-   Verify or edit boxes
-   Progress updates (e.g., 3 of 7 frames verified)

Verification depth is user-controlled.

------------------------------------------------------------------------

# 4. Event Status Model

Event status is derived from file-level verification.

Suggested levels:

-   Representative verified
-   Partially verified (some frames verified)
-   Fully verified (all frames verified)

Or simply:

-   X of Y frames verified

Avoid calling an event fully "verified" unless all files are verified.

------------------------------------------------------------------------

# 5. Detection and Boxes

All detection truth lives at file level.

Each file stores:

-   Bounding boxes
-   Instance labels
-   Verification status
-   Reviewer metadata

Events never overwrite file-level truth.

------------------------------------------------------------------------

# 6. Burst Context Handling

Burst context is shown inside event review:

-   Filmstrip or small grid (max \~9 recommended for quick scanning)
-   Provides temporal and spatial context
-   Supports detecting missed animals or incorrect tracks

User may: - Verify only representative frame - Verify selected
additional frames - Verify all frames

There is no separate burst verification mode.

Burst is contextual support inside event review.

------------------------------------------------------------------------

# 7. Independence Interval Changes

Events are computed views based on user-defined independence rules.

When rules change:

-   Recompute event groupings
-   Reuse existing file-level verification
-   Derive new event statuses automatically

Since verification is stored at file level, no annotation data is lost.

------------------------------------------------------------------------

# 8. Final Architecture Summary

-   Events are review containers.
-   Files are annotation truth objects.
-   Boxes live at file level.
-   Event status is derived.
-   Users may verify only representative frame or all frames.
-   There is a single unified review mode.

This design provides:

-   Stable downstream box data
-   Flexibility for users
-   Robust handling of changing event rules
-   Clear separation between semantic (event) and spatial (box) truth

------------------------------------------------------------------------

End of document.
