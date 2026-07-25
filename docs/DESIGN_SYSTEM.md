# QuantLab Design System

**Status: binding.** PRD Module 13 requires this document to exist before any UI
code, and requires it to constrain that code rather than describe it afterwards.
No component, page, chart or dashboard may introduce a colour, a spacing value, a
font size or a number format that is not defined here.

The tokens are not maintained in this file. They are defined once in
[`scripts/build_design_tokens.py`](../scripts/build_design_tokens.py) and
**generated** into three artifacts that both frontends consume:

| Artifact | Consumed by |
|---|---|
| `config/design-tokens.css` | Next.js and Streamlit |
| `config/tailwind.preset.js` | Next.js |
| `config/plotly-template-dark.json`, `-light.json` | Next.js and Streamlit |

```bash
python scripts/build_design_tokens.py           # regenerate
python scripts/build_design_tokens.py --check   # CI: fail on drift
```

`tests/design/test_design_tokens.py` runs `--check` and re-verifies contrast on
every commit. A hand-edited generated file fails the build. This is the mechanism
that makes "binding" true rather than aspirational — the same reasoning as the
research log in `CLAUDE.md` §3: a rule enforced only by discipline is not enforced.

---

## 1. What this is, and what it is not

Three reference points, in order of influence:

- **Bloomberg Terminal** — information density. A screen is a working surface, not
  a landing page. Twenty rows of a table are more useful than four rows with
  breathing room, because the user's job is comparison.
- **Linear** — restraint. One accent, no gradients, no decorative motion, tight
  typography, nothing that draws the eye away from content.
- **FT / Economist** — chart convention. Direct labels over legends, a single
  horizontal grid, no chart frames, muted benchmark series, and a headline that
  states the finding rather than naming the axes.

**This is not a dark SaaS dashboard.** Specifically forbidden: purple/indigo
accent gradients, glassmorphism, drop shadows on cards, rounded corners above
3px, coloured "hero" panels, illustrative icons beside metrics, animated counters,
and any use of colour that does not encode a value.

The governing rule, from which most of the others follow:

> **Colour encodes meaning. Everything that is not encoding data is neutral.**

A button is neutral. A card border is neutral. A heading is neutral. A number is
coloured only when its sign or its status is the point.

---

## 2. Colour

### 2.1 Neutral scale — one hue family

Both modes are drawn from a single low-chroma cool-grey family. There is no
second neutral, no warm grey, no "slate vs zinc" choice at the component level.

| Token | Dark | Light | Use |
|---|---|---|---|
| `bg` | `#0B0D10` | `#FAFBFC` | Page background |
| `surface` | `#14171C` | `#FFFFFF` | Cards, panels, chart plot area |
| `surface-raised` | `#1B1F26` | `#F1F3F6` | Table headers, hover rows, tooltips |
| `border` | `#262B33` | `#DCE0E6` | Separators, table rules, card edges |
| `border-strong` | `#5A6675` | `#7F8996` | Focus rings, active boundaries |
| `text-primary` | `#E6E9EE` | `#12161B` | Values, headings |
| `text-secondary` | `#A8B2BF` | `#49535F` | Labels, axis titles |
| `text-muted` | `#838D99` | `#636D79` | Units, footnotes, null markers |

### 2.2 Semantic — financial meaning only

| Token | Dark | Light | Encodes |
|---|---|---|---|
| `positive` | `#3FB950` | `#12703A` | Gain, outperformance, pass |
| `negative` | `#FF6B61` | `#B4291F` | Loss, underperformance, fail |
| `warning` | `#E3A008` | `#7A5200` | Unrealistic costs, stale data, caveat |
| `info` | `#58A6FF` | `#0B5FCC` | Train split, neutral annotation |
| `benchmark` | `#8B949E` | `#636D79` | The benchmark series. Always muted grey. |

"Neutral" as a financial state (a flat return, an untested factor) is
`text-secondary` — it is the absence of a signal, so it gets no hue.

**Red and green are never the only channel.** Every signed number carries an
explicit `+` or `−` (§5), which is the redundant encoding for the ~8% of men with
a red-green deficiency. This is why the sign rule is a formatting requirement and
not a stylistic preference.

**Semantic colours are reserved.** They never appear as chart series colours. A
green line in a multi-series chart would compete with green-means-gain everywhere
else in the product.

### 2.3 Verified contrast

Computed by `verify_contrast()`, not asserted. All 52 permitted pairs clear
**WCAG AA**: 4.5:1 for text (including semantic colours, which always carry a
number), 3:1 for `border-strong` as a non-text UI boundary.

**Dark**

| Token | on `bg` | on `surface` | on `surface-raised` |
|---|---|---|---|
| `text-primary` | 15.99 | 14.76 | 13.58 |
| `text-secondary` | 9.07 | 8.37 | 7.70 |
| `text-muted` | 5.78 | 5.33 | 4.91 |
| `positive` | 7.66 | 7.07 | 6.51 |
| `negative` | 6.97 | 6.44 | 5.92 |
| `warning` | 8.62 | 7.95 | 7.32 |
| `info` | 7.70 | 7.11 | 6.54 |
| `benchmark` | 6.33 | 5.84 | 5.37 |
| `border-strong` | 3.33 | — | — |

**Light**

| Token | on `bg` | on `surface` | on `surface-raised` |
|---|---|---|---|
| `text-primary` | 17.53 | 18.16 | 16.34 |
| `text-secondary` | 7.55 | 7.82 | 7.03 |
| `text-muted` | 5.07 | 5.26 | 4.73 |
| `positive` | 5.95 | 6.17 | 5.55 |
| `negative` | 6.18 | 6.41 | 5.76 |
| `warning` | 6.68 | 6.92 | 6.23 |
| `info` | 5.75 | 5.96 | 5.36 |
| `benchmark` | 5.07 | 5.26 | 4.73 |
| `border-strong` | 3.42 | — | — |

`border` is deliberately below 3:1 (1.37 dark / 1.28 light). It is a decorative
separator, never the sole indicator of a boundary or a state; anything load-
bearing uses `border-strong`.

### 2.4 Chart palettes

Three palettes, each for one job. **Never** substitute one for another.

**Categorical — identity.** Five slots, assigned in fixed order and never cycled.
A sixth series folds into "Other", becomes small multiples, or gets a composite
encoding. Five is a deliberate ceiling: QuantLab charts compare three cost models
or a handful of factors, and a chart needing eight colours is a chart that should
have been a table.

| Slot | Dark | Light |
|---|---|---|
| 1 | `#1D6DB3` | `#015EA2` |
| 2 | `#C97F3A` | `#C88244` |
| 3 | `#0FA2A2` | `#15A4A3` |
| 4 | `#934F7D` | `#8C2271` |
| 5 | `#607111` | `#556418` |

Validated — lightness band, chroma floor, all-pairs CVD separation, normal-vision
floor and surface contrast all **PASS** in both modes, with no warnings:

```
dark  · worst all-pairs #934F7D↔#1D6DB3  ΔE 9.1 (protan) · tritan 9.3 · all ≥3:1
light · worst all-pairs #8C2271↔#015EA2  ΔE 9.3 (deutan) · tritan 8.4 · all ≥3:1
```

**Sequential — magnitude.** One hue, monotone lightness. IC decay, quantile
buckets, exposure heatmaps.

| Dark | `#2B5378` `#356B9C` `#3F86C2` `#66A6DA` `#96C5EC` |
|---|---|
| **Light** | `#93BAE4` `#6098D2` `#357BBB` `#175A94` `#0A3D68` |

**Diverging — polarity.** Two poles with a **neutral grey** midpoint, never a hue
at the middle. Correlation matrices, active weights, over/under-performance.

| Dark | `#FF6B61` → `#B4655F` → `#5A6068` → `#4E8A57` → `#3FB950` |
|---|---|
| **Light** | `#B4291F` → `#A56A5C` → `#8C939C` → `#4C7F52` → `#12703A` |

Never a rainbow scale. Never a hue at a diverging midpoint. Never a dual-axis
chart — two measures of different scale become two charts, small multiples, or an
index to a common base.

---

## 3. Typography

Two families, and the boundary between them is a rule, not a preference.

- **Inter** — all UI text: labels, headings, prose, buttons, axis titles.
- **JetBrains Mono** — **all numbers, without exception.** Every price, return,
  ratio, p-value, count, date, run id and percentage.

```css
.ql-num {
  font-family: var(--ql-font-mono);
  font-variant-numeric: tabular-nums;
  font-feature-settings: "tnum" 1, "zero" 1;
}
```

> This single rule does more for the product's credibility than anything else in
> this document. Digits of equal width align into columns, so a reader compares
> magnitudes by scanning down rather than by reading each value. A proportional
> `1` is narrower than a `0`, which silently destroys that alignment and makes a
> table of returns look like marketing copy.

Every number in a table, a metric card, a tooltip, an axis tick and a chart
annotation carries `.ql-num` (or Tailwind `font-mono tabular-nums`). A number in
Inter is a bug.

### Scale

Six steps; size and line height always travel together.

| Step | Size / line height | Use |
|---|---|---|
| `xs` | 11 / 16 | Axis ticks, footnotes, badge text |
| `sm` | 12 / 18 | Table cells, secondary labels |
| `base` | 13 / 20 | Body, form controls, primary table text |
| `md` | 15 / 22 | Card values, section headings |
| `lg` | 18 / 26 | Page headings |
| `xl` | 24 / 32 | The single headline metric on a page |

The base is 13px, not 16px. A research terminal fits more rows per screen than a
marketing page does, and that is the point of it.

---

## 4. Spacing and density

4px base, eight steps: `0 4 8 12 16 24 32 48 64`. Nothing between steps, and
nothing above 64px — a gap that large means the layout is wrong.

Density is a requirement, not a taste:

| Rule | Value |
|---|---|
| Metric card height | **< 120px**, including its label, value and context line |
| Table row height | 28px |
| Table header height | 32px |
| Rows visible without scrolling | **≥ 20** (20 × 28 + 32 = 592px) |
| Sidebar width | 260px, fixed — controls only |
| Card padding | 12px (`space-3`); 16px only for a page-level panel |
| Gap between cards | 8px (`space-2`) |

Layout is **sidebar for controls, main area for data**. Filters, date ranges,
universe pickers and cost-model toggles live in the sidebar. The main area is
never interrupted by a control that could have lived there.

---

## 5. Number formatting

One implementation, shared by both frontends. Formatting divergence between the
Next.js and Streamlit views is a defect, which is why the API returns
display-ready strings alongside raw values (PRD Module 12).

| Quantity | Format | Example |
|---|---|---|
| Return, drawdown | Percentage, 1dp, **explicit sign** | `+12.4%`, `−8.1%` |
| Ratio (Sharpe, Sortino, Calmar, IR) | 2dp | `1.23` |
| Information coefficient | 4dp | `0.0087` |
| p-value | 3dp; scientific below 0.001 | `0.032`, `4.1e-05` |
| t-statistic | 2dp, explicit sign | `+2.41` |
| Currency, large | Abbreviated | `₹1.2M`, `₹840.5K` |
| Currency, exact | Thousands-separated, 2dp | `₹1,204,338.50` |
| Share / quantity count | Thousands-separated, 0dp | `12,480` |
| Date | ISO 8601 | `2021-12-31` |
| Percentage of a whole | 1dp, no sign | `34.2%` |
| **Null / missing** | **Em dash `—`** in `text-muted` | `—` |

Rules that are not negotiable:

- **Negatives use the negative colour and a `−` sign. Never parentheses.**
  Accounting parentheses cost a character of scanning and break digit alignment
  in a tabular-figures column.
- **Null is `—`, never `0`, `NaN`, `null`, `N/A` or an empty cell.** A missing
  Sharpe and a Sharpe of zero are different claims, and rendering both as `0` is
  the UI equivalent of a silent data bug. `NaN` leaking to the screen is a defect
  in the API layer, not a formatting case.
- Sign is applied to the *value*, not the label. `+12.4%` — never `↑ 12.4%`.
- Percentages are stored as fractions and formatted at the boundary. A component
  that receives `0.124` and a component that receives `12.4` is how a 100×
  reporting error happens.

---

## 6. Charts

The Plotly template files carry these settings; this section explains them so a
deviation is recognisable as a deviation.

**Structure**

- **No chart border, no plot frame, no 3D, no shadows, no gradients, no fills
  under lines** (except drawdown, below).
- **Gridlines: a subtle horizontal `y` grid only.** No vertical grid. A vertical
  grid on a time axis adds ink without enabling a comparison anyone makes.
- Axis lines are off; ticks are outside, 4px, in `border`.
- Plot and paper background are both `surface` — a chart is part of its card, not
  a window onto a different surface.
- Margins are tight (`l:56 r:16 t:32 b:40`). A chart is not padded like a photo.

**Series**

- **Direct-label series at the end of the line wherever the geometry allows.**
  This is the FT convention and it is preferred over a legend, because it removes
  the round trip between the key and the mark.
- A legend appears **only** when direct labelling is impossible — more than four
  series, or lines that converge at the right edge. When it appears it is
  horizontal, above the plot, unboxed, in `text-secondary`. `showlegend` defaults
  to `false` precisely so that shipping a legend is a decision.
- Because direct labels are the default and are mandatory whenever a legend is
  absent, **identity is never carried by colour alone** in either case.
- Line width 2px, markers ≥8px, no marker on a dense time series.
- **Series labels wear text tokens, not the series colour.** The coloured mark
  next to the label carries identity; colouring the text too makes small type
  fail contrast.

**Conventions that encode meaning**

- **The benchmark is always `benchmark` grey, always dashed, never a bright
  colour and never a categorical slot.** It is context, not a competitor.
- **Drawdown is always shaded below zero**, filled to the zero line in `negative`
  at reduced opacity, with the y-axis running from the trough to 0.
- Positive/negative bars (monthly returns, active weights) use the semantic
  colours; a single-series bar chart of unsigned magnitudes uses slot 1.
- Every chart has a **table view** available. This is the accessibility fallback
  and it is also how a researcher gets the number rather than the picture.

**Tick formats**

Percentages `,.1%`; currency `,.0f` with thousands separators; dates ISO
(`%Y-%m-%d`). Tick labels render in JetBrains Mono at 11px — axis numbers obey
the tabular-figures rule exactly like table numbers do.

**Interaction**

Crosshair with a unified hover on time series (`hovermode: "x unified"`),
per-mark hover on bars and heatmaps. Tooltip body is mono, on `surface-raised`
with a `border-strong` edge.

---

## 7. Statistical honesty in the UI

This section is a **design requirement**, not an afterthought. `CLAUDE.md` §3 and
PRD Module 13 both specify it, and the component contracts below are what make it
structural rather than a review checklist.

### 7.1 A Sharpe ratio cannot be rendered as a number

The API never returns a bare Sharpe (PRD Module 12), and the UI must not be able
to accept one. `SharpeDisplay` takes the object, mirroring
`quantlab.research.reporting.SharpeReport.as_dict()` field for field:

```ts
/** Mirrors SharpeReport.as_dict(). Every field before the optionals is required. */
export interface SharpeReport {
  readonly value: number;
  readonly trials: number;              // read from research_log, never entered
  readonly deflated_sharpe: number;     // P[true Sharpe > 0] after correction
  readonly expected_max_sharpe: number;
  readonly split: string;
  readonly cost_model: string;
  readonly run_id: string;
  readonly n_periods: number;
  readonly significant: boolean;
  readonly provenance: string;          // "exploratory" | "pre-registered (H1)"
  readonly holdout_touched: boolean;
  readonly pbo: number | null;
}

export function SharpeDisplay(props: { report: SharpeReport }): JSX.Element;
```

There is **no** `value?: number` overload, no `trials?: number`, and no
`SharpeDisplay({ value: 1.8 })`. Passing a bare number is a type error:

```tsx
<SharpeDisplay value={1.8} />              // ✗ does not compile
<SharpeDisplay report={backtest.sharpe} /> // ✓ the only form
```

The rendered output always shows, adjacent and at equal visual weight, the
Sharpe, the trials count and the deflated Sharpe. The deflated value is not
hidden behind a tooltip or an info icon — a reader who sees `1.82` must see
`312 trials · deflated 0.31` without an interaction. When
`significant === false` the component appends the reason in `warning`, because a
deflated Sharpe below the 0.95 threshold is the finding.

The same contract applies to `SharpeReport` in tearsheets and any exported PDF.

### 7.2 Split badge — always present

Every backtest result, factor evaluation, model evaluation and chart derived from
one carries a visible split badge. There is no "no badge" state.

| Split | Colour | Note |
|---|---|---|
| `TRAIN` | `info` | |
| `VALIDATION` | `warning` | |
| `HOLDOUT` | `negative`, filled | The loudest element on the screen |
| `PRE-SAMPLE` | `text-muted` | |

`quantlab.config.resolve_split` returns **compound labels** when a window spans a
boundary — `train+validation`, `pre-sample+train`. The badge must render these as
distinct segments (`TRAIN` `+` `VALIDATION`), never collapse them to the cleaner
of the two, and never fall through to a default. An unrecognised split string
renders as `UNKNOWN` in `negative` rather than as no badge: an unlabelled result
is worse than a wrongly labelled one, because it looks fine.

`HOLDOUT` is filled rather than outlined and is the only filled badge in the
system. The holdout is evaluated once, at the end of the project, and a screen
showing holdout numbers should be visibly unusual.

### 7.3 Unrealistic-cost warning marker

Any result whose `cost_model` is not exactly `IndianEquityCostModel` carries a
`warning` marker — an icon plus the model name, never colour alone.

| `cost_model` | Marker |
|---|---|
| `IndianEquityCostModel` | none |
| `IndianEquityCostModel(turnover-approx)` | ⚠ `turnover approximation` |
| `GenericBpsModel` | ⚠ `flat bps — not Indian statutory costs` |
| `ZeroCostModel` | ⚠ `GROSS — not a tradeable result` |

`ZeroCostModel` results additionally render their Sharpe in `text-muted` rather
than `text-primary`. A gross Sharpe is a validation instrument, and it should not
look like a result.

### 7.4 Exploratory vs pre-registered

`provenance` distinguishes them, and they must be visually distinct at a glance:

- **Pre-registered** — a solid `border-strong` left rule on the result card, and
  the hypothesis id (`H1`) rendered as a link to `docs/hypotheses.md`.
- **Exploratory** — a **dashed** left rule in `text-muted` and the literal word
  `EXPLORATORY`. Never hidden, never abbreviated to an icon.

Most research is exploratory and that is legitimate. The rule exists so an
exploratory result is never presented with the authority of a prediction.

---

## 8. Pages

Nine, per PRD Module 13. Each is sidebar-plus-main and obeys the density rules:
Overview, Data Quality, Universe, Feature Explorer, Factor Research, Backtest,
Research Log, Risk, Portfolio.

The Streamlit dashboard in `dashboard/` consumes the same API and the same
tokens, and is deliberately plainer: it is the researcher-facing tool, where
Next.js is the product-facing one.

---

## 9. Checklist for any UI pull request

- [ ] No literal hex, rgb, px spacing value or font size anywhere in a component
- [ ] Every number renders in JetBrains Mono with `tabular-nums`
- [ ] Every signed number carries an explicit `+` / `−`; negatives are coloured,
      not parenthesised
- [ ] Nulls render as `—`, never `0`, `NaN` or blank
- [ ] Every Sharpe goes through `SharpeDisplay` with a full `SharpeReport`
- [ ] Every result carries a split badge, including compound splits
- [ ] Non-`IndianEquityCostModel` results carry the warning marker
- [ ] Exploratory results are visually distinguished from pre-registered ones
- [ ] Charts: no vertical grid, no frame, no gradient; benchmark grey and dashed
- [ ] Series are direct-labelled, or a legend is present and justified
- [ ] Every chart has a table view
- [ ] Metric cards under 120px; tables show 20 rows without scrolling
- [ ] `python scripts/build_design_tokens.py --check` passes
