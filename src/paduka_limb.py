"""
Paduka - the whole below-knee limb, block-printed.

Everything before this put pattern on a flat plate and read it from above.
The reference does something harder and much better: it wraps a printed
cloth around a leg. Two things follow.

**The mapping.** A pattern painted onto a curved limb by projecting it
straight down smears down every steep face - which is what wrecked the
earlier foot. A leg is very nearly a surface of revolution, so the right
projection is cylindrical: unroll the limb into (angle x height), lay the
repeat out in that flat space, and roll it back. Motifs then sit square to
the surface everywhere, and because the angular pitch is constant they come
out larger on the calf and smaller at the ankle - exactly the way a printed
sock behaves, and exactly what the reference shows.

**The colour is printed, not raised.** In the reference the pattern is
flush with the surface. So the colour here is carried per FACE rather than
as relief bodies: the mesh is dense enough that a face is about a
millimetre across, each face is tested against the repeat in unrolled
space, and the mesh is split into one body per colour. Nothing is extruded,
nothing streaks, and it is a truer description of a block-printed object
anyway - ink has no thickness.

The navy sole and the navy ankle ring are the same mechanism, selected by
height rather than by pattern.
"""

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union
from shapely import affinity
import shapely

import palampore as PM
from palampore import clean, largest, place
from paduka_pattern import buta_outline, rosette
from paduka_leg import loft

# --- the limb ---------------------------------------------------------------
# z, half-width across (y), half-depth (x), centre x offset
# A below-knee socket: calf belly high up, a long taper, a narrow ankle.
LIMB = [
    (430.0, 45.0, 41.0,  0.0),
    (412.0, 47.5, 43.5, -0.5),
    (390.0, 49.5, 45.5, -1.5),
    (366.0, 50.0, 46.0, -2.5),
    (340.0, 48.5, 44.5, -3.0),
    (312.0, 45.5, 42.0, -3.0),
    (284.0, 42.0, 39.0, -2.5),
    (256.0, 38.5, 36.0, -2.0),
    (228.0, 35.0, 33.0, -1.0),
    (200.0, 31.5, 30.0,  0.5),
    (176.0, 28.0, 27.5,  2.0),
    (156.0, 25.0, 25.5,  3.5),
    (140.0, 22.5, 24.0,  4.5),
    (128.0, 21.0, 23.0,  5.0),
    (120.0, 20.5, 22.5,  5.0),
]
ANKLE_Z = 120.0

# the shoe-like foot below the ankle: x station, centre y, half-width,
# ground clearance, top height
FOOT = [
    ( -8.0, -1.0, 10.0, 14.0,  44.0),
    (  0.0, -1.0, 18.0,  6.0,  60.0),
    (  8.0, -1.0, 25.0,  1.0,  78.0),
    ( 18.0, -1.5, 29.0,  0.0,  95.0),
    ( 30.0, -2.0, 31.0,  0.0, 108.0),
    ( 44.0, -2.5, 32.0,  0.0, 116.0),
    ( 58.0, -3.0, 32.0,  0.4, 119.0),
    ( 74.0, -3.5, 31.5,  1.2, 115.0),
    ( 92.0, -3.5, 31.5,  1.8, 105.0),
    (112.0, -3.0, 32.5,  2.0,  93.0),
    (132.0, -2.0, 34.5,  1.8,  80.0),
    (152.0, -0.5, 37.0,  1.2,  67.0),
    (170.0,  1.0, 39.5,  0.6,  56.0),
    (188.0,  2.0, 41.0,  0.2,  46.0),
    # the toe box stays fat and only rounds off over the last 20mm. Taper
    # it any earlier and the foot reads as a cone, which is what the first
    # two attempts did.
    (206.0,  2.5, 41.0,  0.0,  40.0),
    (222.0,  2.5, 40.5,  0.0,  36.5),
    (236.0,  2.0, 39.0,  0.0,  32.5),
    (248.0,  1.5, 36.5,  0.0,  28.5),
    (258.0,  1.0, 33.0,  0.0,  24.0),
    (266.0,  0.5, 27.0,  0.0,  19.0),
    (272.0,  0.0, 18.0,  0.0,  13.0),
    (276.0,  0.0,  8.0,  0.0,   7.0),
]


FOOT_X0 = 26.0        # where the ankle column lands on the foot
N_RING = 220
N_LEVEL = 150

R_REF = 44.0          # the radius the unrolled pattern is scaled against
SOLE_Z = 11.0          # navy sole: everything below this
RING_Z = (122.0, 133.0)   # navy ankle ring

CREAM = np.array([0.940, 0.918, 0.865])
NAVY = np.array([0.098, 0.145, 0.290])
INK = {
    "indigo": np.array([0.145, 0.255, 0.520]),
    "madder": np.array([0.640, 0.180, 0.170]),
    "gold":   np.array([0.780, 0.600, 0.270]),
}


# ---------------------------------------------------------------- geometry ---

def _ring(z, ry, rx, cx, n=N_RING, p=2.35):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([cx + FOOT_X0 + rx * np.sign(c) * np.abs(c) ** (2 / p),
                            ry * np.sign(s) * np.abs(s) ** (2 / p),
                            np.full(n, z)])


def _foot_section(x, cy, w, z0, z1, n=N_RING, p=2.4, flat=1.5, dome=0.88):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = w * np.sign(c) * np.abs(c) ** (2 / p)
    zc, rz = (z0 + z1) / 2, (z1 - z0) / 2
    z = np.where(s >= 0, zc + rz * np.abs(s) ** dome,
                 zc - rz * np.abs(s) ** flat)
    return np.column_stack([np.full(n, x), cy + y, z])


def limb_solid():
    zs = np.array([r[0] for r in LIMB])
    secs = []
    for z in np.linspace(LIMB[0][0], ANKLE_Z, N_LEVEL):
        ry = np.interp(z, zs[::-1], [r[1] for r in LIMB][::-1])
        rx = np.interp(z, zs[::-1], [r[2] for r in LIMB][::-1])
        cx = np.interp(z, zs[::-1], [r[3] for r in LIMB][::-1])
        secs.append(_ring(z, ry, rx, cx))
    m = loft(secs[::-1])
    m.process(validate=True)
    return m


def foot_solid():
    secs = [_foot_section(*s) for s in FOOT]
    m = loft(secs)
    m.process(validate=True)
    return m


def build_limb():
    m = trimesh.boolean.union([limb_solid(), foot_solid()], engine="manifold")
    m = m.subdivide_to_size(2.0)
    m.process(validate=True)
    return m


# ----------------------------------------------------------------- pattern ---

def sprig(scale=1.0, seed=0):
    """
    The blue reference motif: a little spray - a pair of serrated leaves, a
    stem, and two or three open flowers. Drawn once, stamped everywhere.
    """
    rng = np.random.default_rng(seed)
    parts = [PM.ribbon([(0, 0), (1.5 * scale, 9 * scale),
                        (0.5 * scale, 20 * scale)], 1.9 * scale, 0.9 * scale,
                       taper=0.9, n=60)]
    for k, (f, side) in enumerate(((0.30, 1), (0.46, -1), (0.62, 1),
                                   (0.76, -1))):
        L = (11.0 - 1.4 * k) * scale
        bl, _v = PM.leaf(L, L * 0.44, teeth=5, curl=0.3)
        parts.append(place(bl, (0.9 + 1.2 * f) * scale, 20 * scale * f,
                           rot=52 * side + 40, flip=side < 0))
    for f, side, d in ((0.52, 1, 9.5), (0.74, -1, 8.0), (0.95, 1, 10.5)):
        ring, eye = PM.sprig(d * scale, petals=7, rot=float(rng.random()))
        g = unary_union([ring, eye])
        parts.append(place(g, (1.0 + 3.4 * side) * scale, 21 * scale * f))
    return clean(unary_union(parts))


def buta_motif(scale=1.0, seed=0):
    """the paisley reference motif: an outlined buta with a filled interior"""
    w = 17.0 * scale
    outer = buta_outline(w)
    rim = clean(outer.difference(outer.buffer(-1.6 * scale)))
    inner = largest(clean(outer.buffer(-(3.4 * scale))))
    veins = []
    if not inner.is_empty:
        b = inner.bounds
        for k in range(5):
            f = 0.22 + 0.62 * k / 4
            y = b[1] + (b[3] - b[1]) * f
            veins.append(Point((b[0] + b[2]) / 2, y).buffer(1.5 * scale, 20))
        rim = clean(unary_union([rim, inner.difference(
            inner.buffer(-1.3 * scale))]))
    return rim, clean(unary_union(veins)) if veins else Polygon()


def repeat(motif_fn, pitch_u, pitch_v, u0, u1, v0, v1, jitter=0.0, seed=1):
    """a half-drop repeat of one stamp across the unrolled surface"""
    rng = np.random.default_rng(seed)
    out = []
    for i, u in enumerate(np.arange(u0, u1, pitch_u)):
        drop = (pitch_v / 2) if i % 2 else 0.0
        for v in np.arange(v0 + drop, v1, pitch_v):
            g = motif_fn(seed=int(rng.integers(0, 9999)))
            g = affinity.rotate(g, float(rng.uniform(-jitter, jitter)),
                                origin=(0, 0))
            out.append(affinity.translate(g, u, v))
    return clean(unary_union(out))


# ------------------------------------------------------------------- paint ---

def face_uv(mesh):
    """
    Unroll: every face centroid gets a coordinate in the flat pattern space.

    Above the ankle that is (angle x height) about the limb axis. Below it
    the surface has turned over into a foot, so it switches to the foot's
    own plane - length along the foot, and distance out from its
    centreline. The seam between the two lands under the navy ankle ring.
    """
    c = mesh.triangles.mean(axis=1)
    u = np.empty(len(c))
    v = np.empty(len(c))

    hi = c[:, 2] >= ANKLE_Z
    th = np.arctan2(c[hi, 1], c[hi, 0] - FOOT_X0)
    u[hi] = th * R_REF
    v[hi] = c[hi, 2]

    lo = ~hi
    # walk the foot: length becomes v, girth becomes u
    u[lo] = c[lo, 1] * 1.05
    v[lo] = ANKLE_Z - (c[lo, 0] - FOOT_X0) * 0.92
    return np.column_stack([u, v])


def paint(mesh, pattern, extra=None):
    """
    Split the mesh into one body per colour by testing each face against
    the unrolled repeat. No geometry is changed - ink has no thickness.
    """
    uv = face_uv(mesh)
    pts = shapely.points(uv)
    groups = {}
    taken = np.zeros(len(uv), dtype=bool)

    for name, geom in (extra or {}).items():
        sel = geom(mesh) & ~taken
        groups[name] = sel
        taken |= sel

    for name, geom in pattern.items():
        if geom.is_empty:
            continue
        shapely.prepare(geom)      # 385k point-in-polygon tests need an index
        sel = shapely.contains(geom, pts) & ~taken
        groups[name] = sel
        taken |= sel

    groups["ground"] = ~taken
    out = {}
    for name, sel in groups.items():
        if sel.sum() == 0:
            continue
        out[name] = mesh.submesh([np.where(sel)[0]], append=True)
    return out


def sole_sel(mesh):
    c = mesh.triangles.mean(axis=1)
    return c[:, 2] < SOLE_Z


def ring_sel(mesh):
    c = mesh.triangles.mean(axis=1)
    return (c[:, 2] >= RING_Z[0]) & (c[:, 2] <= RING_Z[1])


def colourway(name="indigo"):
    """the two colourways in the reference: blue on cream, madder and gold"""
    span_u = (-np.pi * R_REF - 40, np.pi * R_REF + 40)
    span_v = (-140, 460)
    if name == "indigo":
        # Big stamps. The mesh resolves about 2.5mm, so a motif whose
        # details are finer than that comes out as gravel - and the
        # reference motifs are large anyway, a 55mm spray on the calf.
        ink = repeat(lambda seed: sprig(2.4, seed), 80.0, 104.0,
                     *span_u, *span_v, jitter=10.0, seed=3)
        return {"indigo": ink}
    rim_all, vein_all = [], []
    rng = np.random.default_rng(5)
    for i, u in enumerate(np.arange(span_u[0], span_u[1], 80.0)):
        drop = 51.0 if i % 2 else 0.0
        for v in np.arange(span_v[0] + drop, span_v[1], 102.0):
            rim, vein = buta_motif(2.0, seed=int(rng.integers(0, 9999)))
            a = float(rng.uniform(-8, 8))
            rim_all.append(affinity.translate(affinity.rotate(rim, a, origin=(0, 0)), u, v))
            vein_all.append(affinity.translate(affinity.rotate(vein, a, origin=(0, 0)), u, v))
    return {"madder": clean(unary_union(rim_all)),
            "gold": clean(unary_union(vein_all))}


if __name__ == "__main__":
    import time
    t0 = time.time()
    m = build_limb()
    print(f"limb {len(m.faces)} faces in {time.time()-t0:.0f}s"
          f"   bbox {np.round(m.bounds[1]-m.bounds[0], 0)}")
    m.export("limb.stl")
