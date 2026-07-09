"""Validated chart palette (see the `dataviz` skill's `references/palette.md`).

Fixed categorical hue order -- assigned once per entity and never
recolored on filter/sort, per the skill's "color follows the entity"
rule. Validated colorblind-safe via `scripts/validate_palette.js`
(worst adjacent CVD delta-E 24.2, well above the >=12 target) before
use here.
"""

from __future__ import annotations

# Categorical slots 1-5 (blue, aqua, yellow, green, violet), in the
# fixed order the palette reference specifies.
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]

# Slots below 3:1 contrast on the light surface (aqua, yellow) --
# the "relief rule": these must always carry a visible direct label,
# never rely on color alone.
LOW_CONTRAST_SLOTS = {"#1baf7a", "#eda100"}

# Sequential single-hue (blue) ramp, light -> dark, for ordered/magnitude
# encoding (e.g. vocab size order in the cost/benefit scatter).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
DEEMPHASIS_GRAY = "#c3c2b7"
