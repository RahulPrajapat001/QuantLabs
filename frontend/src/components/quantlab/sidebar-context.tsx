"use client";

/**
 * Lets a page put its filters in the sidebar without the shell knowing what any
 * page's controls are.
 *
 * The design system's rule is that controls live in the sidebar and the main
 * area holds only data (§4). Pages are siblings of the sidebar in the DOM, so
 * obeying that rule needs a portal; without one, every page would drift a date
 * picker into its own header and the rule would decay into a comment.
 */

import { useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";

/** The id of the host the shell renders inside the `<aside>`. */
export const SIDEBAR_HOST_ID = "ql-sidebar-controls";

/** Renders `children` into the sidebar. A no-op during SSR. */
export function SidebarPortal({ children }: { children: React.ReactNode }) {
  const [host, setHost] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setHost(document.getElementById(SIDEBAR_HOST_ID));
  }, []);

  if (!host) return null;
  return createPortal(
    <div className="flex flex-col gap-3 border-t border-border pt-3">{children}</div>,
    host,
  );
}

/** A labelled control. The label binds to the input via a generated id. */
export function Control({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  /** Receives the id to put on the actual form control. */
  children: (id: string) => React.ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <label
        htmlFor={id}
        className="text-xs uppercase tracking-wide text-text-secondary"
      >
        {label}
      </label>
      {children(id)}
      {hint && <span className="text-xs text-text-muted">{hint}</span>}
    </div>
  );
}
