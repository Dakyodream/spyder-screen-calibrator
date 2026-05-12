"""
SpyderCheckr 24 reference CIE Lab values (D50 illuminant).

Sources:
  - Datacolor SpyderCheckr 24 official reference data
  - Values represent the ideal colorimetric targets under D50

Patch layout (physical card, PORTRAIT orientation — 4 cols × 6 rows):
  Reading order: left→right, top→bottom

       E          F           G             H
  1  White      Cyan       Orange      Bluish Green
  2  Neutral8  Magenta   Purplish Blue  Blue Flower
  3  Neutral6.5 Yellow   Moderate Red   Foliage
  4  Neutral5   Red        Purple       Blue Sky
  5  Neutral3.5 Green    Yellow Green  Light Skin
  6  Black      Blue     Orange Yellow  Dark Skin
"""

# CIE Lab D50 reference values for the 24 color patches
# Order: portrait reading order (left→right, top→bottom), 4 cols × 6 rows
# Format: (name, L*, a*, b*)
COLOR_PATCHES: list[tuple[str, float, float, float]] = [
    # Row 1:  E          F            G             H
    ("White",            96.54,  -0.43,   1.19),   # 1E
    ("Cyan",             51.04, -28.63, -28.64),   # 1F
    ("Orange",           62.66,  36.07,  57.10),   # 1G
    ("Bluish Green",     70.72, -33.40,  -0.20),   # 1H
    # Row 2
    ("Neutral 8",        81.26,  -0.64,  -0.34),   # 2E
    ("Magenta",          51.94,  49.99, -14.57),   # 2F
    ("Purplish Blue",    40.02,  10.41, -45.96),   # 2G
    ("Blue Flower",      55.11,   8.84, -25.40),   # 2H
    # Row 3
    ("Neutral 6.5",      66.77,  -0.73,  -0.50),   # 3E
    ("Yellow",           81.73,   4.04,  79.82),   # 3F
    ("Moderate Red",     51.12,  48.24,  16.25),   # 3G
    ("Foliage",          43.14, -13.10,  21.91),   # 3H
    # Row 4
    ("Neutral 5",        50.87,  -0.15,  -0.27),   # 4E
    ("Red",              42.10,  53.38,  28.19),   # 4F
    ("Purple",           30.33,  22.98, -21.59),   # 4G
    ("Blue Sky",         49.93,  -4.88, -21.93),   # 4H
    # Row 5
    ("Neutral 3.5",      35.66,  -0.42,  -1.23),   # 5E
    ("Green",            55.26, -38.34,  31.37),   # 5F
    ("Yellow Green",     72.53, -23.71,  57.26),   # 5G
    ("Light Skin",       65.71,  18.13,  17.81),   # 5H
    # Row 6
    ("Black",            20.46,   0.07,  -0.46),   # 6E
    ("Blue",             28.78,  14.18, -50.30),   # 6F
    ("Orange Yellow",    71.94,  19.36,  67.86),   # 6G
    ("Dark Skin",        37.99,  13.56,  14.06),   # 6H
]

# Grayscale strip — 8 steps from near-black to near-white
# Format: (name, L*, a*, b*)
GRAY_PATCHES: list[tuple[str, float, float, float]] = [
    ("Gray 1 (darkest)", 20.46,   0.07,  -0.46),
    ("Gray 2",           29.50,  -0.10,  -0.30),
    ("Gray 3",           38.50,  -0.20,  -0.40),
    ("Gray 4",           49.50,  -0.15,  -0.27),
    ("Gray 5",           57.00,  -0.20,  -0.35),
    ("Gray 6",           66.77,  -0.73,  -0.50),
    ("Gray 7",           80.00,  -0.50,  -0.40),
    ("Gray 8 (lightest)",96.54,  -0.43,   1.19),
]

# Combined ordered list used for matrix computation
ALL_PATCHES = COLOR_PATCHES  # + GRAY_PATCHES

# sRGB display values for each patch — portrait order (4 cols × 6 rows).
# Values measured from the physical SpyderCheckr 24 card.
COLOR_PATCHES_SRGB: list[tuple[str, int, int, int]] = [
    # Row 1:  E              F              G              H
    ("White",          249, 242, 238),   # 1E
    ("Cyan",             0, 127, 159),   # 1F
    ("Orange",         222, 118,  32),   # 1G
    ("Bluish Green",    98, 187, 166),   # 1H
    # Row 2
    ("Neutral 8",      202, 192, 195),   # 2E
    ("Magenta",        192,  75, 145),   # 2F
    ("Purplish Blue",   58,  88, 159),   # 2G
    ("Blue Flower",    126, 125, 174),   # 2H
    # Row 3
    ("Neutral 6.5",    161, 157, 154),   # 3E
    ("Yellow",         245, 205,   0),   # 3F
    ("Moderate Red",   195,  79,  95),   # 3G
    ("Foliage",         82, 106,  60),   # 3H
    # Row 4
    ("Neutral 5",      122, 118, 116),   # 4E
    ("Red",            186,  26,  51),   # 4F
    ("Purple",          83,  58, 106),   # 4G
    ("Blue Sky",        87, 120, 155),   # 4H
    # Row 5
    ("Neutral 3.5",     80,  80,  78),   # 5E
    ("Green",           57, 146,  64),   # 5F
    ("Yellow Green",   157, 188,  54),   # 5G
    ("Light Skin",     197, 145, 125),   # 5H
    # Row 6
    ("Black",           43,  41,  43),   # 6E
    ("Blue",            25,  55, 135),   # 6F
    ("Orange Yellow",  238, 158,  24),   # 6G
    ("Dark Skin",      112,  76,  60),   # 6H
]

GRAY_PATCHES_SRGB: list[tuple[str, int, int, int]] = [
    ("Gray 1",  52,  52,  52),
    ("Gray 2",  80,  80,  80),
    ("Gray 3", 105, 105, 105),
    ("Gray 4", 133, 133, 133),
    ("Gray 5", 152, 152, 152),
    ("Gray 6", 160, 160, 160),
    ("Gray 7", 200, 200, 200),
    ("Gray 8", 243, 243, 243),
]

# Neutral patches used for WB calibration: 4 brightest neutral grays only.
# Dark patches (Neutral 3.5, Black) are excluded — backlight bleed
# contaminates their measurements and would skew the correction.
_NEUTRAL_INDICES = [0, 4, 8, 12]  # White, Neutral 8, Neutral 6.5, Neutral 5
NEUTRAL_PATCHES_SRGB: list[tuple[str, int, int, int]] = [
    COLOR_PATCHES_SRGB[i] for i in _NEUTRAL_INDICES
]
