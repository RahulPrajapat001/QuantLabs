import * as React from "react";

import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => (
    <input
      ref={ref}
      type={type}
      className={cn(
        "h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary",
        "placeholder:text-text-muted focus-visible:border-border-strong ql-fade",
        // A date input holds digits, and digits are mono everywhere.
        (type === "date" || type === "number") && "ql-num",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export { Input };
