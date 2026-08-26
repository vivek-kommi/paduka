"""
Paduka - pattern language module.

2D contour generation for the carved relief:
  hashia  : border band following the true sole curve
  zanjeer : chain of studs inboard of the border
  buta    : hooked teardrop (the motif that became Paisley), drawn as a
            DOUBLE outline with an interior of rosettes and paired leaves
  buti    : smaller scattered motifs filling the gaps

Everything is a real 2D contour (shapely), so the relief that gets extruded
from it is actual solid geometry, not a displacement map.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from shapely import affinity

# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------

def closed_spline(pts, n=600, smooth=0.0):
    """Periodic cubic spline through pts -> closed polyline (n,2)."""
    p = np.asarray(pts, dtype=float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, per=True, k=3)
    u = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def open_spline(pts, n=300, smooth=0.0):
    p = np.asarray(pts, dtype=float)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, per=False, k=3)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def clean(g):
    if g.is_empty:
        return g
    g = g.buffer(0)
    return g


# ----------------------------------------------------------------------------
# the sole outline - a proper foot curve, not an ellipse
# ----------------------------------------------------------------------------
# x runs heel(0) -> toe(260). +y is medial (big-toe side), -y is lateral.
# Targets: L 260, ball 95, waist 62, heel 68.

FOOT_PTS = [
    (  0.5,   1.0),   # heel back, medial of centre
    (  7.0,  17.0),
    ( 18.0,  28.0),
    ( 34.0,  33.5),
    ( 58.0,  33.0),
    ( 84.0,  29.0),
    (105.0,  26.6),   # waist, medial
    (128.0,  28.0),
    (152.0,  34.0),
    (172.0,  41.5),
    (188.0,  45.4),   # 1st metatarsal head - widest medial
    (206.0,  45.0),
    (224.0,  41.0),
    (240.0,  33.0),
    (252.0,  21.0),
    (259.0,   6.0),   # toe
    (258.0,  -6.0),
    (250.0, -18.0),
    (238.0, -27.5),
    (222.0, -36.0),
    (203.0, -44.0),
    (186.0, -49.6),   # 5th metatarsal head - widest lateral
    (170.0, -50.0),
    (152.0, -46.0),
    (130.0, -39.5),
    (108.0, -35.4),   # waist, lateral
    ( 84.0, -33.6),
    ( 58.0, -34.0),
    ( 34.0, -34.5),
    ( 18.0, -29.0),
    (  6.0, -16.0),
    (  0.0,  -1.0),
]


def sole_outline(n=900):
    c = closed_spline(FOOT_PTS, n=n)
    poly = Polygon(c).buffer(0)
    # normalise length to exactly 260 and shift so heel-back = x0
    minx, miny, maxx, maxy = poly.bounds
    poly = affinity.scale(poly, xfact=260.0 / (maxx - minx), yfact=1.0,
                          origin=(minx, 0))
    poly = affinity.translate(poly, xoff=-poly.bounds[0])
    return poly


def width_at(poly, x):
    minx, miny, maxx, maxy = poly.bounds
    seg = LineString([(x, miny - 10), (x, maxy + 10)]).intersection(poly)
    return 0.0 if seg.is_empty else seg.length


def centre_at(poly, x):
    minx, miny, maxx, maxy = poly.bounds
    seg = LineString([(x, miny - 10), (x, maxy + 10)]).intersection(poly)
    if seg.is_empty:
        return 0.0
    return seg.centroid.y


# ----------------------------------------------------------------------------
# buta - the hooked teardrop
# ----------------------------------------------------------------------------
# Built from a spine curve plus a half-width profile, capped with a round
# bulbous base and converging to the hooked tip. Normalised to height 1.0.
#
# JUDGEMENT CALLS (reconstructed from description, not photographs):
#   width : height           = 0.60
#   hook  : tip crosses back over the spine by ~0.22 of the height
#   base  : bulb half-width  = 0.30 of height  (i.e. bulb dia 0.60 h)

SPINE = [
    (0.000, 0.000),
    (0.005, 0.180),
    (0.030, 0.360),
    (0.085, 0.520),
    (0.170, 0.660),
    (0.272, 0.775),
    (0.362, 0.868),
    (0.418, 0.952),
    (0.404, 1.022),
    (0.330, 1.048),
    (0.252, 1.020),
    (0.216, 0.958),
]


# proportion judgements (reconstructed, see summary):
#   BULB_X   where the bulb centreline sits across the motif width
#   the overall height/width ratio emerges from the spine at about 1.7
BULB_X = 0.36


def _buta_halfwidth(s):
    """half-width factor along arclength s in [0,1]."""
    # bulbous low down, slightly pinched at the very base, tapering to
    # nothing at the hooked tip
    w = (1.0 - s) ** 0.58
    w *= 1.0 + 0.14 * np.sin(np.pi * np.clip(s, 0, 1) ** 1.25)
    w *= 1.0 - 0.22 * np.exp(-((s / 0.16) ** 2))
    return w


def buta_outline(width, n=460):
    """Solid hooked-teardrop silhouette, `width` across at its widest,
    base sitting on y=0, bulb centreline on x=0."""
    sp = open_spline(SPINE, n=n)
    d = np.diff(sp, axis=0)
    s = np.concatenate([[0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    s = s / s[-1]

    t = np.gradient(sp, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
    nrm = np.column_stack([-t[:, 1], t[:, 0]])

    hw = 0.30 * _buta_halfwidth(s)
    left = sp + nrm * hw[:, None]
    right = sp - nrm * hw[:, None]

    r = hw[0]
    ang = np.linspace(np.pi, 2 * np.pi, 48)
    cap = np.column_stack([sp[0, 0] + r * np.cos(ang), sp[0, 1] + r * np.sin(ang)])
    ring = np.vstack([left[::-1], cap, right])

    poly = Polygon(ring).buffer(0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    b = poly.bounds
    k = width / (b[2] - b[0])
    poly = affinity.scale(poly, xfact=k, yfact=k, origin=(0, 0))
    b = poly.bounds
    return affinity.translate(poly, -b[0] - width * BULB_X, -b[1])


def rosette(cx, cy, r, petals=6, rot=0.0):
    """small bold flower - a lobed disc, legible at arm's length."""
    a = np.linspace(0, 2 * np.pi, 220, endpoint=False)
    rr = r * (0.86 + 0.14 * np.cos(petals * (a - rot)))
    return Polygon(np.column_stack([cx + rr * np.cos(a), cy + rr * np.sin(a)]))


def leaf(cx, cy, length, width, angle):
    """almond / lens shaped leaf."""
    t = np.linspace(0, 1, 70)
    x = (t - 0.5) * length
    y = width * 0.5 * np.sin(np.pi * t) ** 0.80
    ring = np.vstack([np.column_stack([x, y]),
                      np.column_stack([x[::-1], -y[::-1]])])
    p = Polygon(ring).buffer(0)
    p = affinity.rotate(p, angle, origin=(0, 0))
    return affinity.translate(p, cx, cy)


def _core_spine(core, n=40):
    """sample the medial line of a polygon by horizontal scanlines."""
    cb = core.bounds
    out = []
    for y in np.linspace(cb[1] + 0.5, cb[3] - 0.5, n):
        seg = LineString([(cb[0] - 5, y), (cb[2] + 5, y)]).intersection(core)
        if seg.is_empty:
            continue
        if seg.geom_type != "LineString":
            seg = max(seg.geoms, key=lambda g: g.length)
        out.append((seg.centroid.x, y, seg.length))
    return out


def buta_motif(width, wall=4.6, gap=2.6, inner_wall=3.2,
               mode="ring", eye=True):
    """
    Full buta: a DOUBLE outline - an outer ring with a second concentric
    contour inside - with the interior carrying small rosettes, paired
    leaves and a drainage eye rather than being left solid.

    mode="ring"   second contour is a ring, interior filled with ornament
    mode="island" second contour is a solid teardrop island (used where the
                  motif is too small to carry a ring and a filling both)
    mode="single" one outline only

    Returns (raised_geometry, [through_holes]).
    """
    outer = buta_outline(width)
    parts = [outer.difference(outer.buffer(-wall))]
    holes = []

    def largest(g):
        if g.is_empty:
            return g
        if isinstance(g, MultiPolygon):
            return max(g.geoms, key=lambda q: q.area)
        return g

    if mode == "single":
        void = largest(outer.buffer(-(wall + 1.5)))
    else:
        inner_body = largest(outer.buffer(-(wall + gap)))
        if inner_body.is_empty or inner_body.area < 30:
            return clean(unary_union(parts)), holes
        if mode == "island":
            parts.append(inner_body)
            void = inner_body.buffer(-1.0)     # ornament sits ON the island
            island = True
        else:
            parts.append(inner_body.difference(inner_body.buffer(-inner_wall)))
            void = largest(inner_body.buffer(-(inner_wall + 1.4)))
            island = False
        if mode == "island":
            void = largest(void)

    if void.is_empty or void.area < 18:
        return clean(unary_union([p for p in parts if not p.is_empty])), holes

    sp = [p for p in _core_spine(void, 56) if p[2] > 2.0]
    if not sp:
        return clean(unary_union([p for p in parts if not p.is_empty])), holes

    dy = max(0.4, sp[1][1] - sp[0][1]) if len(sp) > 1 else 1.0

    # --- the eye: a genuine drainage perforation seated in the bulb ------
    eye_r, eye_y = 0.0, None
    lower = sp[: max(3, int(len(sp) * 0.55))]
    widest = max(lower, key=lambda p: p[2])
    if eye and widest[2] >= 6.4:
        r = min(2.9, widest[2] * 0.36)
        if r >= 2.5:
            holes.append(Point(widest[0], widest[1]).buffer(r, 64))
            eye_r, eye_y = r, widest[1]

    if mode == "island":
        # the island is solid relief; ornament on it would not read, so it
        # carries only the eye
        raised = clean(unary_union([p for p in parts if not p.is_empty]))
        return raised, holes

    # --- rosettes and paired leaves packed up the interior ---------------
    def run(stations, k0=0):
        k, i = k0, 0
        while i < len(stations) - 1:
            cx, cy, w = stations[i]
            if k % 2 == 0:
                r = min(3.3, w * 0.40)
                if r < 1.9:
                    i += 1
                    continue
                parts.append(rosette(cx, cy, r, petals=6, rot=0.55 * k))
                adv = (2 * r + 2.0) / dy
            else:
                ll = min(w * 0.95, 8.5)
                ww = max(2.3, min(3.4, w * 0.34))
                if ll < 3.8 or w < 4.8:
                    i += 1
                    continue
                parts.append(leaf(cx - w * 0.15, cy, ll, ww, 52))
                parts.append(leaf(cx + w * 0.15, cy, ll, ww, -52))
                adv = (ww + 2.4) / dy
            i += max(2, int(np.ceil(adv)))
            k += 1
        return k

    if eye_y is not None:
        below = [p for p in sp if p[1] < eye_y - eye_r - 2.2][::-1]
        above = [p for p in sp if p[1] > eye_y + eye_r + 2.2]
        run(below, 0)
        run(above, 1)
    else:
        run(sp, 0)

    raised = clean(unary_union([p for p in parts if not p.is_empty]))
    return raised, holes


def buti_motif(width, wall=4.2):
    """smaller filler motif - single bold outline, rosette heart, eye."""
    return buta_motif(width, wall=wall, mode="single", eye=True)


def place(geom, x, y, scale=1.0, rot=0.0, mirror=False):
    g = geom
    if mirror:
        g = affinity.scale(g, xfact=-1, yfact=1, origin=(0, 0))
    if scale != 1.0:
        g = affinity.scale(g, xfact=scale, yfact=scale, origin=(0, 0))
    if rot:
        g = affinity.rotate(g, rot, origin=(0, 0))
    return affinity.translate(g, x, y)
