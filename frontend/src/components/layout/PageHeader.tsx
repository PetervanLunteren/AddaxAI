/**
 * The header at the top of a page: its title, a one-line subtitle, and
 * optional page-level buttons at the right.
 *
 * One component so the spacing and the place of the buttons are the same
 * on every page that uses it. The Insights pages do; new pages should
 * too, instead of writing the markup out again.
 */

interface PageHeaderProps {
  title: string;
  subtitle: string;
  /** Page-level buttons, e.g. Download CSV. */
  actions?: React.ReactNode;
}

export function PageHeader({ title, subtitle, actions }: PageHeaderProps) {
  return (
    <header className="border-b bg-card/80 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
            <p className="text-sm text-muted-foreground">{subtitle}</p>
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      </div>
    </header>
  );
}
