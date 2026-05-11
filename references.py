"""
SpyderCheckr 24 reference CIE Lab values (D50 illuminant).

Sources:
  - Datacolor SpyderCheckr 24 official reference data
  - Values represent the ideal colorimetric targets under D50

Patch layout (physical card, front face - 24 color patches):
  Row 0 (top):    0  1  2  3  4  5
  Row 1:          6  7  8  9 10 11
  Row 2:         12 13 14 15 16 17
  Row 3 (bot):   18 19 20 21 22 23

Grayscale strip (back face or bottom strip, 8 steps from dark to light):
  GS0 .. GS7
"""

# CIE Lab D50 reference values for the 24 color patches
# Format: (name, L*, a*, b*)
COLOR_PATCHES: list[tuple[str, float, float, float]] = [
    # Row 0
    ("Dark Skin",        37.99,  13.56,  14.06),
    ("Light Skin",       65.71,  18.13,  17.81),
    ("Blue Sky",         49.93,  -4.88, -21.93),
    ("Foliage",          43.14, -13.10,  21.91),
    ("Blue Flower",      55.11,   8.84, -25.40),
    ("Bluish Green",     70.72, -33.40,  -0.20),
    # Row 1
    ("Orange",           62.66,  36.07,  57.10),
    ("Purplish Blue",    40.02,  10.41, -45.96),
    ("Moderate Red",     51.12,  48.24,  16.25),
    ("Purple",           30.33,  22.98, -21.59),
    ("Yellow Green",     72.53, -23.71,  57.26),
    ("Orange Yellow",    71.94,  19.36,  67.86),
    # Row 2
    ("Blue",             28.78,  14.18, -50.30),
    ("Green",            55.26, -38.34,  31.37),
    ("Red",              42.10,  53.38,  28.19),
    ("Yellow",           81.73,   4.04,  79.82),
    ("Magenta",          51.94,  49.99, -14.57),
    ("Cyan",             51.04, -28.63, -28.64),
    # Row 3
    ("White",            96.54,  -0.43,   1.19),
    ("Neutral 8",        81.26,  -0.64,  -0.34),
    ("Neutral 6.5",      66.77,  -0.73,  -0.50),
    ("Neutral 5",        50.87,  -0.15,  -0.27),
    ("Neutral 3.5",      35.66,  -0.42,  -1.23),
    ("Black",            20.46,   0.07,  -0.46),
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
ALL_PATCHES = COLOR_PATCHES + GRAY_PATCHES

# sRGB display values for each patch used when rendering the on-screen target.
# These are the standard sRGB values (gamma-encoded, 0-255) corresponding to
# the Lab references above, for display on an uncalibrated sRGB screen.
# Source: X-Rite / Datacolor published sRGB values.
COLOR_PATCHES_SRGB: list[tuple[str, int, int, int]] = [
    ("Dark Skin",       115,  82,  68),
    ("Light Skin",      194, 150, 130),
    ("Blue Sky",        98, 122, 157),
    ("Foliage",         87, 108,  67),
    ("Blue Flower",    133, 128, 177),
    ("Bluish Green",    103, 189, 170),
    ("Orange",          214, 126,  44),
    ("Purplish Blue",   80,  91, 166),
    ("Moderate Red",   193,  90,  99),
    ("Purple",          94,  60, 108),
    ("Yellow Green",   157, 188,  64),
    ("Orange Yellow",  224, 163,  46),
    ("Blue",            56,  61, 150),
    ("Green",           70, 148,  73),
    ("Red",            175,  54,  60),
    ("Yellow",         231, 199,  31),
    ("Magenta",        187,  86, 149),
    ("Cyan",            8, 133, 161),
    ("White",          243, 243, 242),
    ("Neutral 8",      200, 200, 200),
    ("Neutral 6.5",    160, 160, 160),
    ("Neutral 5",      122, 122, 121),
    ("Neutral 3.5",     85,  85,  85),
    ("Black",           52,  52,  52),
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
