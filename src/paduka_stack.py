"""
Paduka - the stack: a modular below-knee limb that slots onto the paduka.

Three things this adds to the single-piece limb.

**It comes apart.** A prosthesis is not one object, it is a stack of
components, and a printed one should be honest about that. The limb is cut
into three printed modules - foot, shank, cuff - joined by an oval spigot
and a flared ferrule. The spigot is the limb's own cross-section scaled
down and squashed, so it keys against rotation without a single added
feature: the anti-rotation detail IS the anatomy. The ferrule flares
outward as it rises, so every module prints without support at its joint
and leaves a 2mm shoulder at each seam - the brass ferrules of a
nineteenth-century wooden leg, doing a job rather than being quoted.

**It slots onto the paduka.** The foot module is the anatomical foot, with
the dovetail channel and the toe-post cleft from `paduka_modular` cut into
its underside. The same foot drops onto any paduka in the wardrobe, and the
khadau post between the great toe and the second is still the locating pin.

**The pattern is denser and it knows where the joints are.** A tighter
half-drop of large stamps, a second offset grid of buti between them, and a
printed band at every seam. The bands are not decoration bolted on - they
fall exactly where the modules part, so the object tells you how it comes
apart by how it is printed, which is what a garment's cuffs and plackets do.
"""

import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely import affinity
import shapely

import palampore as PM
from palampore import clean, largest, place
from paduka_pattern import rosette
from paduka_leg import loft
import paduka_pattern as PP
import paduka_foot as PF
import paduka_modular as MOD
import paduka_limb as LB
import paduka_ankle as AK

# --- one frame --------------------------------------------------------------
# paduka_modular works in the sole frame: sole x 0..260, foot at x=7 on a
# 14mm plate. paduka_limb works in its own. Reconcile once, here.
PLATE_T = MOD.PLATE_T
SEAT = np.array([MOD.FOOT_X, 0.0, PLATE_T])
AXIS_X = MOD.FOOT_X + PF.ANKLE_X     # the limb axis, over the talus
AXIS_Y = -3.0
ANKLE_TOP = 96.0 + PLATE_T           # where the anatomical ankle column ends
LIMB_DZ = ANKLE_TOP - LB.ANKLE_Z     # so the limb's ankle lands on the foot's

LIMB = LB.LIMB
N_RING = 240
N_LEVEL = 175
MESH_MM = 1.75                       # the pattern cannot resolve finer

# --- the joints (sandal frame) ---------------------------------------------
JOINT = (106.0, 270.0)
SPIG_H = 20.0
SPIG_FX, SPIG_FY = 0.62, 0.48        # squashed, therefore keyed
FERRULE = 2.2
FERR_H = 8.5
CLEAR = 0.32
PIN_D = 4.4
TOP_Z = LIMB[0][0] + LIMB_DZ

# --- pattern ----------------------------------------------------------------
R_REF = 44.0
UV_ANKLE = 86.0                      # where the unrolling changes chart.
                                     # It has to sit BELOW the lowest limb
                                     # band, or that band gets cut in a
                                     # sawtooth where the chart hands over
FOOT_ZC = PLATE_T + 24.0             # the axis of the foot, treated as a tube
FOOT_R = 38.0        # not the foot's true radius: it is chosen so the
                     # repeat runs at nearly the same angular pitch as the
                     # limb's, or the pattern visibly changes gauge at the ankle
FOOT_V0 = -230.0                     # the foot's own stretch of the repeat,
                                     # clear of every seam band
WELT_Z = PLATE_T + 12.0



# ---------------------------------------------------------------- geometry ---

def _ring(z, ry, rx, cx, n=N_RING, p=2.35):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.column_stack([
        AXIS_X + cx + rx * np.sign(c) * np.abs(c) ** (2 / p),
        AXIS_Y + ry * np.sign(s) * np.abs(s) ** (2 / p),
        np.full(n, z)])


def limb_solid():
    """
    The shank and calf, in the sandal frame. Its lowest section is carried
    18mm BELOW the top of the anatomical ankle so the two solids interpenetrate
    and the union is a blend rather than a butt joint.
    """
    zs = np.array([r[0] for r in LIMB])[::-1]
    ry = np.array([r[1] for r in LIMB])[::-1]
    rx = np.array([r[2] for r in LIMB])[::-1]
    cx = np.array([r[3] for r in LIMB])[::-1]
    secs = []
    for z in np.linspace(LIMB[0][0], LB.ANKLE_Z - 18.0, N_LEVEL):
        secs.append(_ring(z + LIMB_DZ,
                          float(np.interp(z, zs, ry)),
                          float(np.interp(z, zs, rx)),
                          float(np.interp(z, zs, cx))))
    m = loft(secs[::-1]); m.process(validate=True)
    return m


def foot_solid():
    """the anatomical foot with its second pass of anatomy, on the plate"""
    f = PF.detailed_foot()
    f.apply_translation(SEAT)
    return f


def whole_limb():
    m = trimesh.boolean.union([foot_solid(), limb_solid()], engine="manifold")
    return m


# ------------------------------------------------------------------ joints ---

def section_poly(mesh, z):
    sl = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    p2, _T = sl.to_2D(to_2D=np.eye(4))
    return max(p2.polygons_full, key=lambda g: g.area)


def _polar(poly, n=200):
    """
    r(theta) about the centroid. Sampling both rings of a loft on the SAME
    theta array is the whole trick: lofting two independently-walked
    exteriors twists them into a bowtie, which is what the first cut did.
    """
    c = np.array(poly.centroid.coords[0])
    p = np.array(poly.exterior.coords[:-1]) - c
    a = np.arctan2(p[:, 1], p[:, 0])
    r = np.hypot(p[:, 0], p[:, 1])
    o = np.argsort(a)
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return c, th, np.interp(th, a[o], r[o], period=2 * np.pi)


def _tube(c, th, r0, r1, z0, z1, sx0=1.0, sy0=1.0, sx1=1.0, sy1=1.0):
    a = np.column_stack([c[0] + sx0 * r0 * np.cos(th),
                         c[1] + sy0 * r0 * np.sin(th), np.full(len(th), z0)])
    b = np.column_stack([c[0] + sx1 * r1 * np.cos(th),
                         c[1] + sy1 * r1 * np.sin(th), np.full(len(th), z1)])
    m = loft([a, b]); m.process(validate=True)
    return m


def spigot(poly, z, h=SPIG_H, grow=0.0):
    """the plug: the section squashed to an oval, tapered 1mm over its height"""
    c, th, r = _polar(poly)
    return _tube(c, th, r + grow / SPIG_FX, r + (grow - 1.0) / SPIG_FX,
                 z - 0.8, z + h,
                 SPIG_FX, SPIG_FY, SPIG_FX, SPIG_FY)


def ferrule(poly, z, h=FERR_H, out=FERRULE):
    """the collar: flush below, standing proud at the seam, flaring upward"""
    c, th, r = _polar(poly)
    return _tube(c, th, r, r + out, z - h, z)


def pin_cut(poly, z, d=PIN_D):
    c = np.array(poly.centroid.coords[0])
    m = trimesh.creation.cylinder(radius=d / 2, height=200, sections=24)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0]))
    m.apply_translation([c[0], c[1], z + SPIG_H * 0.55])
    return m


def paduka_cuts():
    """the interface to the sandal, lifted straight out of paduka_modular"""
    sole = PP.sole_outline()
    cy = PP.centre_at(sole, 124.0)
    rail = MOD.dovetail(MOD.RAIL_W, MOD.RAIL_H,
                        MOD.RAIL_X1 - MOD.RAIL_X0 + 70, MOD.RAIL_X0 - 35,
                        cy, clearance=MOD.CLEAR)
    rail.apply_translation([0, 0, PLATE_T - 0.01])
    post = MOD.toe_post(clearance=MOD.CLEAR)
    slot = trimesh.creation.box(extents=[80, MOD.POST_D + 2 * MOD.CLEAR, 70])
    slot.apply_translation([MOD.POST_X + 32, MOD.POST_Y, PLATE_T + 26])
    return [rail, post, slot]


RAW_NAMES = ("foot", "shank", "cuff")
NAMES = ("foot", "ankle", "shank", "cuff")


def modules(whole=None):
    from trimesh.intersections import slice_mesh_plane
    whole = whole if whole is not None else whole_limb()
    polys = [section_poly(whole, z) for z in JOINT]

    parts, lo = [], whole
    for z in JOINT:
        parts.append(slice_mesh_plane(lo, [0, 0, -1], [0, 0, z], cap=True))
        lo = slice_mesh_plane(lo, [0, 0, 1], [0, 0, z], cap=True)
    parts.append(lo)

    out = {}
    for i, (nm, m) in enumerate(zip(RAW_NAMES, parts)):
        add, cut = [], []
        if i < len(JOINT):
            z, p = JOINT[i], polys[i]
            add += [spigot(p, z), ferrule(p, z)]
            cut.append(pin_cut(p, z))
        if i > 0:
            z, p = JOINT[i - 1], polys[i - 1]
            cut += [spigot(p, z, h=SPIG_H + 0.6, grow=CLEAR), pin_cut(p, z)]
        if nm == "foot":
            cut += paduka_cuts()
        if add:
            m = trimesh.boolean.union([m] + add, engine="manifold")
        # one cut at a time: batching them into a single difference loses the
        # manifold engine on this geometry. And NO process(validate=True)
        # afterwards - it merges vertices in a way that punches holes in the
        # freshly-cut caps.
        for c in cut:
            m = trimesh.boolean.difference([m, c], engine="manifold")
        out[nm] = m

    # the foot comes apart again at the malleoli: a single-axis ankle. See
    # paduka_ankle - the pin goes where the malleoli are, because that is
    # what the malleoli ARE.
    foot, ankle = AK.split(out["foot"], AXIS_X, AXIS_Y)
    out["foot"], out["ankle"] = foot, ankle
    return {k: out[k] for k in NAMES}


# ----------------------------------------------------------------- pattern ---

import paduka_paisley as PZ

# Four filaments. The ground is the wood-fibre filament; the other three are
# the outline, fill and detail blocks of a Sanganeri print, in that order.
WAYS = {
    "madder": dict(ground=np.array([0.948, 0.906, 0.812]),
                   outline=np.array([0.075, 0.098, 0.165]),
                   fill=np.array([0.660, 0.170, 0.140]),
                   detail=np.array([0.800, 0.625, 0.290])),
    "indigo": dict(ground=np.array([0.958, 0.941, 0.898]),
                   outline=np.array([0.070, 0.115, 0.255]),
                   fill=np.array([0.160, 0.300, 0.590]),
                   detail=np.array([0.955, 0.945, 0.925])),
}

U_SPAN = (-165.0, 165.0)
V_SPAN = (-470.0, 480.0)
SEAM_V = tuple(z - LIMB_DZ for z in JOINT) + (TOP_Z - LIMB_DZ,)


# The foot gets its own stretch of the repeat, and a quieter one. A shoe
# upper is always plainer than the cloth it is cut from: it is a smaller,
# more curved, more crowded surface, and a repeat dense enough to look right
# on a calf turns to gravel on a set of toes. So the field below the ankle
# is a wider half-drop with no ground dotting and no fillers, the toes are
# left bare, and two hashiya bands - one round the heel, one across the ball
# - do the work that density does higher up.
FOOT_V = (-392.0, -168.0)     # instep and heel; below this are the toes
FIELD_V0 = -172.0             # the dense field starts here and runs up

# --- why this is banded -----------------------------------------------------
# An all-over repeat on a 420mm object is beautiful and unprintable on an
# AMS. Colour changes are not charged per motif, they are charged per LAYER:
# any layer that contains two filaments costs a tool change and a purge, and
# an all-over repeat puts every filament in every one of ~2100 layers. That
# is of the order of 1.9kg of purge for 2.1kg of object.
#
# Banding is the only structural fix - the ornament is gathered into
# horizontal bands, and the layers between them are single-filament and free.
# It is also how the textile it comes from is actually made up into a
# garment: a plain field with worked borders at the hem, the cuff and the
# seam. So the bands land on the module seams, and the object tells you
# where it comes apart by where it is ornamented.
BANDED = True
# One worked band per module, placed by hand rather than by a pitch. A
# pitch always ends up putting half a band off the end of something; naming
# the two positions takes ten seconds and cannot.
LIMB_BANDS = [(228.0, 64.0),      # the shank, mid-calf
              (392.0, 64.0)]      # the cuff
SEAM_H = 14.0                 # the rules that mark a joint - no butas

# The two charts have two different circumferences, and a band has to close
# on itself in whichever one it lives in. u = theta * R, so the seam down the
# back of the leg sits at u = +/- pi*R - a different number above and below
# the ankle. Laying every band out across its OWN circumference is what stops
# a motif being sliced in half at the seam.
LIMB_U = np.pi * R_REF
LIMB_N = 7                    # butas round the leg
FOOT_N = 6                    # butas round the foot


def band_zones():
    """the v ranges that carry a worked band, in unrolled space"""
    return [(v - h, v) for v, h in LIMB_BANDS]


# Where the foot wears its borders. These are planes across the foot's
# LENGTH, so each one reads as a strap: one over the instep just in front of
# the ankle, one across the ball at the toe break. Between and beyond them
# the foot is plain, which is what a shoe upper is.
# --- how much of this is worth printing -------------------------------------
# The foot is printed sole-down, so a band ACROSS it is a vertical plane and
# lands on every layer between the sole and the instep. Measured: the vamp
# band and the toe rule put ink on 82% of the foot's layers, for 10g of ink
# and 213g of purge. Nothing about that is a settings problem.
#
# SIMPLE gathers the whole answer:
#   - the limb drops to TWO filaments. The fill block printed 1.4g on the
#     foot and flushed 60g. It was a rounding error paid for in kilos.
#   - the foot keeps only its welt, which is the one ornament on it that
#     lies flat. Everything else moves to the leg, where a band is
#     horizontal and costs its own height and nothing more.
#   - band butas become solid silhouettes rather than double outlines.
# The paduka is untouched: it is 44mm tall, so its polychrome is free.
SIMPLE = True

FOOT_BANDS = [] if SIMPLE else [(-250.0, 62.0)]
FOOT_RULE = None if SIMPLE else (-380.0, 14.0)

# The welt - the dark sole panel - is drawn in the DRAWING, as a strip of
# the foot's own chart, not selected by height as it used to be. A height
# threshold can only ever be as smooth as the triangles that carry it, and
# at any mesh a printer would tolerate that edge comes out visibly stepped.
# As a strip in u it is a panel of the pattern, so its edge is as smooth as
# the drawing, and it follows the form instead of cutting across it.
FOOT_WELT_U = (-88.0, -30.0) if SIMPLE else (-98.0, -20.0)
FOOT_WELT_V = (-390.0, -184.0)    # heel to toe break; the toes stay bare


def colourway(name="madder", pitch=(54.0, 72.0), w_big=33.0):
    """
    The limb's drawing, cut as TWO blocks rather than three.

    The detail block is gone, and what it used to draw is left as bare
    ground showing through the fill - which is not a compromise, it is what
    a resist print is: the pattern you see is the dye that did NOT reach the
    cloth. The ground dotting has gone with it.

    Three reasons, and only the first is about looks.

      it is cleaner    the dotting was the busiest thing on the object and
                       it was fighting the butas at arm's length
      it is a filament every layer of a 420mm object that carries pattern
                       carries a tool change per colour in it. Going from
                       three inks to two takes a third of those changes -
                       and the purge with them - straight out of the job
      it is register   two blocks need one register gap, not three, so the
                       drawing can be finer at the same nozzle

    The paduka keeps all four filaments. It is 44mm tall instead of 420, so
    its polychrome costs almost nothing, and the shoe should be the jewel.
    """
    O, F = [], []

    def band(v, h, u_half, n, seed, dotted=True, solid=False):
        o, f, d = PZ.hashiya(v, -u_half, u_half, height=h, count=n,
                             seed=seed, dotted=dotted, solid=solid)
        # the border's rules and dots were the detail block; with the detail
        # block gone they join the outline
        O.append(clean(unary_union([o, d])))
        F.append(f)

    # --- the limb: seam borders, and a worked band between each pair -------
    # the seam borders sit just BELOW their joint, so no band runs onto a
    # spigot or a collar face and gets held at the ground filament halfway
    # through a motif
    for k, v in enumerate(SEAM_V):
        band(v - 3.0, SEAM_H, LIMB_U, LIMB_N, k + 1, dotted=False)
    for k, (v, h) in enumerate(LIMB_BANDS):
        band(v, h, LIMB_U, LIMB_N, k + 21, solid=SIMPLE)

    # --- the foot: a vamp band and a toe-break rule, on the foot's own
    #     circumference, which is a different number from the limb's --------
    fu = np.pi * FOOT_R
    for k, (v, h) in enumerate(FOOT_BANDS):
        band(v, h, fu, FOOT_N, k + 41)
    if FOOT_RULE:
        band(FOOT_RULE[0], FOOT_RULE[1], fu, FOOT_N, 51, dotted=False)

    # the welt, cut back wherever a band crosses it: a motif half swallowed
    # by a solid panel reads as a printing fault, which is the one thing
    # this drawing must never look like
    w = box(FOOT_WELT_U[0], FOOT_WELT_V[0], FOOT_WELT_U[1], FOOT_WELT_V[1])
    for a, h in FOOT_BANDS + ([FOOT_RULE] if FOOT_RULE else []):
        w = clean(w.difference(box(-1e4, a - h - PZ.REG, 1e4, a + PZ.REG)))
    O.append(w)

    out = {"outline": clean(unary_union([g for g in O if not g.is_empty]))}
    if not SIMPLE:
        out["fill"] = clean(unary_union([g for g in F if not g.is_empty]))
    return out


def face_uv(c):
    """
    Two charts, both cylindrical, because both parts of the object are
    tubes - they just lie along different axes.

    Above the ankle the tube is vertical: unroll about the limb axis.
    Below it the tube lies fore-and-aft: unroll about the FOOT's axis, so
    the repeat wraps the foot the way a sock does. The earlier version
    projected the foot straight down, which smeared every stamp down the
    medial and lateral faces - the same failure the flat plate had.

    The foot is given its own stretch of the repeat, well clear of the seam
    bands, so a band drawn at a height on the limb cannot reappear as a
    stripe across the instep.
    """
    u, v = np.empty(len(c)), np.empty(len(c))
    hi = c[:, 2] >= UV_ANKLE
    u[hi] = np.arctan2(c[hi, 1] - AXIS_Y, c[hi, 0] - AXIS_X) * R_REF
    v[hi] = c[hi, 2] - LIMB_DZ
    lo = ~hi
    u[lo] = np.arctan2(c[lo, 2] - FOOT_ZC, c[lo, 1] - AXIS_Y) * FOOT_R
    v[lo] = FOOT_V0 - (c[lo, 0] - AXIS_X)
    return np.column_stack([u, v])


# --- the surfaces nobody ever sees ------------------------------------------
# Every one of these faces is buried when the stack is together: the spigot
# inside its socket, the annulus a ferrule seats against, the flat of a seam,
# the dovetail channel and post cleft under the foot. Printing pattern on
# them costs tool changes, purge and time for something that is by definition
# invisible - so they are held at the ground filament and the drawing simply
# stops at the seam line, which is what a tailor does at a facing.
SOCKET_D = SPIG_H + 1.4


def _norm_r(c, poly):
    """each face's horizontal radius about a joint, as a fraction of the
    limb's own section there - so one number covers an oval"""
    ctr, th, r = _polar(poly, 128)
    a = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
    rr = np.interp(a, th, r, period=2 * np.pi)
    return np.hypot(c[:, 0] - ctr[0], c[:, 1] - ctr[1]) / np.maximum(rr, 1e-6)


def buried(c, polys):
    """faces inside a joint, or on the flat of one"""
    z = c[:, 2]
    sel = np.zeros(len(c), bool)
    for j, p in zip(JOINT, polys):
        # The spigot and the socket bore occupy the SAME volume - one on the
        # module below the seam, one on the module above - so a single test
        # catches both, and it has to be bounded by radius or it would swallow
        # everything above the joint on the upper module.
        r = _norm_r(c, p)
        sel |= (z >= j - 0.05) & (z <= j + SOCKET_D) & (r < 0.86)
        sel |= np.abs(z - j) < 0.45          # the seam flat and collar face
    # under the foot: the sole flat, the dovetail channel, the post cleft
    sel |= z <= PLATE_T + 0.35
    rail_cy = PP.centre_at(PP.sole_outline(), 124.0)
    sel |= ((z <= PLATE_T + MOD.RAIL_H + 0.6)
            & (np.abs(c[:, 1] - rail_cy) <= MOD.RAIL_W / 2 + 2.0)
            & (c[:, 0] >= MOD.RAIL_X0 - 36) & (c[:, 0] <= MOD.RAIL_X1 + 36))
    sel |= ((np.abs(c[:, 1] - MOD.POST_Y) <= MOD.POST_D / 2 + 2.0)
            & (c[:, 0] >= MOD.POST_X - 11) & (z <= PLATE_T + 34))
    sel |= AK.hidden(c, AXIS_X, AXIS_Y)
    return sel


def paint(mesh, pattern, polys=None):
    c = mesh.triangles.mean(axis=1)
    pts = shapely.points(face_uv(c))
    hide = buried(c, polys if polys is not None else _joint_polys())
    taken = hide.copy()
    groups = {}
    for nm, g in pattern.items():
        if g.is_empty:
            continue
        shapely.prepare(g)
        s = shapely.contains(g, pts) & ~taken
        groups[nm] = groups.get(nm, np.zeros(len(c), bool)) | s
        taken |= s
    groups["ground"] = ~taken | hide
    return {k: mesh.submesh([np.where(v)[0]], append=True)
            for k, v in groups.items() if v.sum()}


_POLY = []


def _joint_polys():
    if not _POLY:
        w = whole_limb()
        _POLY.extend(section_poly(w, z) for z in JOINT)
    return _POLY


_PAT = {}


def drawing(way):
    """the repeat is identical for all three modules - draw it once"""
    import os, pickle
    if way in _PAT:
        return _PAT[way]
    f = f"_pat_{way}{'_band' if BANDED else ''}.pkl"
    if os.path.exists(f):
        with open(f, "rb") as fh:
            _PAT[way] = pickle.load(fh)
        return _PAT[way]
    g = colourway(way)
    with open(f, "wb") as fh:
        pickle.dump(g, fh)
    _PAT[way] = g
    return g


def painted(way="madder", mods=None):
    """every module split into one body per dye, ready for a 3MF or a render"""
    mods = mods or modules()
    pat = drawing(way)
    polys = _joint_polys()
    out = {}
    for nm, m in mods.items():
        d = m.subdivide_to_size(MESH_MM)
        out[nm] = paint(d, pat, polys)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    mods = modules()
    print(f"cut in {time.time()-t0:.0f}s")
    tot = 0
    for k in NAMES:
        m = mods[k]
        e = m.bounds[1] - m.bounds[0]
        tot += m.volume / 1000
        print(f"  {k:6} tris {len(m.faces):7d}  watertight {m.is_watertight!s:5}"
              f"  bbox {e[0]:5.0f} x {e[1]:4.0f} x {e[2]:5.0f}"
              f"  vol {m.volume/1000:6.1f} cm3")
        m.export(f"stack_{k}.stl")
    print(f"  total {tot:.0f} cm3   overall height {TOP_Z:.0f} mm")
