# Plot legends and useful axis ticks

Issue: [#103](https://github.com/ahrefs/ocannl/issues/103)

## Current state

`PrintBox_utils.plot` is still a thin wrapper over `printbox-ext-plot` 0.12.
Series are identified only by their glyphs, and the upstream renderer labels
only axis endpoints. Multi-series plots such as `moons_demo` therefore require
guesswork.

## Goal

Make terminal plots self-explanatory while preserving existing call sites.

## Direction

- Add an optional legend channel to `PrintBox_utils.plot`, separate from the
  upstream `plot_spec list`. A simple `(glyph, label)` list is enough and keeps
  old calls source-compatible.
- Add bounded intermediate ticks in `printbox-ext-plot` itself, where the
  coordinate-to-cell mapping already lives. Reimplementing the upstream axes
  by scraping or wrapping its rendered box would duplicate layout logic and
  be fragile.
- Keep tick density a rendering decision based on plot size. Do not expose a
  large tick-configuration API before a real caller needs it.

## Completion criteria

- A labeled multi-series plot renders a compact legend; unlabeled plots keep
  their current layout.
- Both axes show readable intermediate values without collisions on the
  standard plot sizes.
- `moons_demo` (or a focused fixture) snapshots the legend and ticks.
- Existing plotting call sites still compile, and affected expectations are
  promoted.

If an upstream `printbox-ext-plot` release cannot be made promptly, land the
legend independently and leave ticks blocked on the upstream change rather
than forking its renderer into OCANNL.
