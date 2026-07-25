import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * A card is neutral. No shadow, no gradient, 3px radius, `border` edge.
 * Padding is 12px; a page-level panel may use 16px and nothing else does.
 */
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("border border-border bg-surface rounded", className)}
      {...props}
    />
  ),
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      "flex items-baseline justify-between gap-2 px-3 pt-3 pb-2",
      className,
    )}
    {...props}
  />
));
CardHeader.displayName = "CardHeader";

/**
 * A section heading, not a headline. `md` is the largest a card title gets —
 * `lg` is for page headings and `xl` for the single headline metric on a page.
 */
const CardTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn("text-md font-medium text-text-primary", className)}
    {...props}
  />
));
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-text-secondary", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("px-3 pb-3", className)} {...props} />
));
CardContent.displayName = "CardContent";

export { Card, CardContent, CardDescription, CardHeader, CardTitle };
