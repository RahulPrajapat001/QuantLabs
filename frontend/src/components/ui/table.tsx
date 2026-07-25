import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Density is the requirement: 28px rows, a 32px header, and at least 20 rows
 * visible without scrolling. Twenty rows of a table are more useful than four
 * with breathing room, because the reader's job is comparison.
 */
const Table = React.forwardRef<
  HTMLTableElement,
  React.HTMLAttributes<HTMLTableElement>
>(({ className, ...props }, ref) => (
  <table
    ref={ref}
    className={cn("w-full border-collapse text-sm", className)}
    {...props}
  />
));
Table.displayName = "Table";

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead ref={ref} className={cn("bg-surface-raised", className)} {...props} />
));
TableHeader.displayName = "TableHeader";

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody ref={ref} className={className} {...props} />
));
TableBody.displayName = "TableBody";

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      "h-table-row border-b border-border hover:bg-surface-raised ql-fade",
      className,
    )}
    {...props}
  />
));
TableRow.displayName = "TableRow";

export interface TableHeadProps
  extends React.ThHTMLAttributes<HTMLTableCellElement> {
  /** Right-align. Every numeric column, so digits line up down the column. */
  numeric?: boolean;
}

const TableHead = React.forwardRef<HTMLTableCellElement, TableHeadProps>(
  ({ className, numeric, ...props }, ref) => (
    <th
      ref={ref}
      className={cn(
        "h-table-header px-2 text-left align-middle text-xs font-medium text-text-secondary",
        numeric && "text-right",
        className,
      )}
      {...props}
    />
  ),
);
TableHead.displayName = "TableHead";

export interface TableCellProps
  extends React.TdHTMLAttributes<HTMLTableCellElement> {
  /** Mono, tabular, right-aligned. Every cell holding a number. */
  numeric?: boolean;
}

const TableCell = React.forwardRef<HTMLTableCellElement, TableCellProps>(
  ({ className, numeric, ...props }, ref) => (
    <td
      ref={ref}
      className={cn(
        "px-2 align-middle text-text-primary",
        numeric && "ql-num text-right",
        className,
      )}
      {...props}
    />
  ),
);
TableCell.displayName = "TableCell";

export { Table, TableBody, TableCell, TableHead, TableHeader, TableRow };
