/**
 * Shared per-step heading for the folder-run stepper.
 *
 * Sits naked above each step's body content (not inside a Card) so
 * the wide canvas steps (verify grid, dashboard, save preview) and
 * the narrow form steps share the same visual anchor. Matches the
 * font sizes the old page-level header used.
 */

interface StepHeaderProps {
  title: string;
  caption: string;
}

export function StepHeader({ title, caption }: StepHeaderProps) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
      <p className="mt-1 text-sm text-muted-foreground">{caption}</p>
    </div>
  );
}
