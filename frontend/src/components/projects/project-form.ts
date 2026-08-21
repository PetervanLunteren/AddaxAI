/**
 * Form pieces the project dialogs share.
 *
 * Create, edit and duplicate each build their own zod schema, and each
 * declared the description field itself. Three copies of one line meant
 * three answers to "what does an empty box mean", which is what made
 * clearing a description on the edit dialog silently do nothing.
 *
 * **DuplicateProjectDialog deliberately does not use this**, and that is
 * a dependency problem rather than a decision. It types its form from
 * the schema (`z.infer`) instead of from an API payload type, so a field
 * that transforms makes the form's input and output types differ, and
 * `@hookform/resolvers` resolves its own copy of the react-hook-form
 * types: TypeScript then reports two `Resolver` types "with this name"
 * that "are unrelated", and no combination of `z.input` / `z.output` /
 * context generic reconciles them. Create and edit escape it only
 * because they type their forms as `ProjectCreate` / `ProjectUpdate` and
 * cast the resolver. Duplicate normalises in its submit handler instead
 * (`data.description?.trim() || null`), which reaches the same value, so
 * there is nothing to fix there. Do not "tidy" it into this field
 * without budgeting for the resolver typing first.
 */

import { z } from "zod";

/**
 * The project description field.
 *
 * An empty box becomes `null`, never `undefined`. That distinction is the
 * whole point: `JSON.stringify` drops undefined keys, so the field never
 * reached the server at all, and `update_project` treats a key it was not
 * sent as "leave this one alone" (`model_dump(exclude_unset=True)` in
 * `api/crud/project.py`). Correct PATCH semantics on the server, and a
 * request that could never express "clear it" on the client. Null is the
 * value the nullable column wants and it says the thing out loud.
 *
 * Whitespace only counts as empty, so a box holding three spaces clears
 * the description rather than storing them. The duplicate dialog already
 * trimmed; now all three do.
 */
export const projectDescriptionField = z
  .string()
  .max(500, "Description too long")
  .optional()
  .transform((val) => val?.trim() || null);
