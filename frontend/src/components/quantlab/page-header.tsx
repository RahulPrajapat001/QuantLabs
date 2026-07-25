import { cn } from "@/lib/utils";

/**
 * `lg` heading, one line of context, nothing else. No breadcrumb trail, no
 * hero panel, no icon — the page name is the page name.
 */
export function PageHeader({
  title,
  description,
  actions,
  className,
}: {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn("mb-4 flex items-start justify-between gap-4", className)}
    >
      <div className="flex min-w-0 flex-col gap-1">
        <h1 className="text-lg font-medium text-text-primary">{title}</h1>
        {description && (
          <p className="max-w-prose text-sm text-text-secondary">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
}
