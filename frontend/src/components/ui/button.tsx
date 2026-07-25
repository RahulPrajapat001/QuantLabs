import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * A button is neutral (DESIGN_SYSTEM.md §1). There is no primary/accent variant
 * because an accent-coloured button would be colour that encodes nothing.
 * Emphasis comes from position and from `border-strong`, not from hue.
 *
 * `destructive` is the single exception and it is not decoration: it marks an
 * action whose result is a loss of data, which is a meaning `negative` encodes.
 */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1 rounded border whitespace-nowrap text-sm font-medium ql-fade disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "border-border bg-surface-raised text-text-primary hover:border-border-strong",
        outline:
          "border-border bg-transparent text-text-secondary hover:text-text-primary hover:border-border-strong",
        ghost:
          "border-transparent bg-transparent text-text-secondary hover:text-text-primary hover:bg-surface-raised",
        destructive:
          "border-border bg-surface-raised text-negative hover:border-border-strong",
      },
      size: {
        // Heights are spacing tokens: 24px (space-5) and 32px (space-6).
        sm: "h-5 px-2",
        default: "h-6 px-3",
        icon: "h-6 w-6 px-0",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
