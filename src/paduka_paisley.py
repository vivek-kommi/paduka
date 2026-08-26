"""
Paduka - the paisley blocks.

A block-printed cloth is not one drawing. It is a stack of separately
carved blocks stamped in register, and the order is always the same:

  rekh    the outline block - the line that defines every shape
  datta   the fill block - the colour that sits inside the line
  gad     the detail block - the last pass, the one that makes it expensive

That is exactly a three-filament print, so the drawing here is separated
the same way rather than being drawn as a picture and sliced afterwards.
Each motif function returns (outline, fill, detail) with the register gap
already taken out, which means nothing has to be cut apart later and the
0.9mm ground line between two dyes is a property of the drawing.

The field is built from four blocks at four scales, which is what makes a
Sanganeri cloth look bottomless close up:

  buta    the large hooked teardrop, double-outlined, ornamented inside
  buti    the same motif at 55%, dropped into the gaps facing the other way
  phool   a small rosette in the remaining holes
  chownk  the ground dotting - a fine diaper of dots over everything left

and a border - a hashiya - at every seam.
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from shapely import affinity
import shapely

from paduka_pattern import buta_outline, rosette, leaf as _leaf, _core_spine
from palampore import clean, largest

# These two numbers decide whether the drawing survives the slicer, and the
# first cut of them was too optimistic by about a third.
#
# On a VERTICAL wall an inlay is only as wide as the motif, and the slicer
# lays perimeters 0.42mm wide. A 1.2mm line is under three of them, so the
# slicer fits one, sometimes none, and the motif comes out as a dotted line.
# A 0.9mm gap between two dyes is barely two, so the ground line closes up
# and the two colours smear into each other.
REG = 1.25         # the register gap: three extrusions of clear ground
MIN_W = 1.7        # and four across the thinnest line in the drawing


def _lg(g):
    return largest(g) if not g.is_empty else g


def _ok(g, w=1.55, area=3.0):
    """drop anything the nozzle cannot lay down cleanly"""
    if g.is_empty:
        return g
    g = clean(g.buffer(-w / 2).buffer(w / 2).intersection(g))
    if g.is_empty:
        return g
    ps = g.geoms if g.geom_type == "MultiPolygon" else [g]
    keep = [p for p in ps if p.area >= area]
    return clean(unary_union(keep)) if keep else Polygon()


# ------------------------------------------------------------------- buta ---

def buta_blocks(w=30.0, seed=0, spray=True, hashiya=True):
    """
    One buta, cut as three blocks.

      outline   the rim, plus a second concentric contour inside it, plus a
                ring of dots outside - the doubled line is what separates a
                Sanganeri buta from a clip-art teardrop
      fill      the body between the second contour and the ornament
      detail    the spray climbing the interior - rosettes, paired leaves,
                a rosette seated in the bulb - and the dotting on the rim
    """
    rng = np.random.default_rng(seed)
    outer = buta_outline(w)
    # thin rings, fat body. The fill block is the colour of the motif, so
    # every millimetre spent on outline is taken off the thing you actually
    # see from two metres away.
    # The doubled outline costs wall + gap + inner_wall + register off every
    # side - about 4.7mm. Below about 22mm across there is no interior left
    # to double and the motif shatters into hooks, so small butas get ONE
    # ring and a solid body. The threshold matters more than it looks: a row
    # that crosses it has some butas doubled and some not, and reads as a
    # mistake rather than as a rhythm.
    wall = max(2.1, 0.072 * w)
    gap = max(REG, 0.045 * w)
    iw = max(1.9, 0.046 * w)
    double = w >= 26.0

    rim = clean(outer.difference(outer.buffer(-wall)))
    body = _lg(clean(outer.buffer(-(wall + gap))))
    if body.is_empty or body.area < 20:
        return _ok(rim), Polygon(), Polygon()

    if double:
        ring2 = clean(body.difference(body.buffer(-iw)))
        void = _lg(clean(body.buffer(-(iw + REG))))
    else:
        ring2 = Polygon()
        void = body
        spray = False

    det = []
    if hashiya and w >= 18:
        # the dots that run outside the outline block
        ring = outer.buffer(0.55 * wall).exterior
        n = max(8, int(ring.length / (0.115 * w)))
        for i in range(n):
            p = ring.interpolate(ring.length * i / n)
            det.append(Point(p.x, p.y).buffer(max(1.35, 0.028 * w), 14))

    fill = void
    if spray and not void.is_empty and void.area > 20:
        sp = [p for p in _core_spine(void, 56) if p[2] > 1.5]
        k = 0
        i = 0
        dy = max(0.4, (sp[1][1] - sp[0][1])) if len(sp) > 1 else 1.0
        while i < len(sp) - 1:
            cx, cy, ww = sp[i]
            # the ornament has to stay SMALL. The fill block is the colour
            # of the motif; if the spray eats the void the buta reads as an
            # outline with confetti in it instead of a coloured teardrop.
            if k % 3 == 2 and ww > 0.26 * w:
                ll = min(ww * 0.80, 0.22 * w)
                lw = max(MIN_W, min(0.070 * w, ww * 0.26))
                det.append(_leaf(cx - ww * 0.20, cy, ll, lw, 54))
                det.append(_leaf(cx + ww * 0.20, cy, ll, lw, -54))
                adv = (lw + 2 * REG + 3.2) / dy
            else:
                r = min(0.072 * w, ww * 0.34)
                if r < MIN_W:
                    i += 1
                    continue
                det.append(rosette(cx, cy, r, petals=6, rot=0.5 * k
                                   + float(rng.random())))
                adv = (2 * r + 2 * REG + 3.0) / dy
            i += max(2, int(np.ceil(adv)))
            k += 1
        spray_g = clean(unary_union([d for d in det if not d.is_empty]))
        inside = clean(spray_g.intersection(void))
        fill = clean(void.difference(inside.buffer(REG)))

    outline = _ok(clean(unary_union([rim, ring2])) if double else rim)
    detail = _ok(clean(unary_union([d for d in det if not d.is_empty])),
                 area=1.4)
    return outline, _ok(fill), detail


def phool(d=10.0, seed=0):
    """the small rosette that fills what the butas leave"""
    rng = np.random.default_rng(seed)
    r = rosette(0, 0, d / 2, petals=int(rng.integers(6, 9)),
                rot=float(rng.random()))
    eye = Point(0, 0).buffer(max(MIN_W / 2, d * 0.15), 20)
    return _ok(clean(r.difference(eye.buffer(REG)))), _ok(eye, area=1.2)


# ----------------------------------------------------------------- hashiya ---

def hashiya(v, u0, u1, height=15.0, count=8, seed=0, dotted=True,
            spray=False, solid=False):
    """
    The border: a heavy rule, a run of butas standing on it, a dotted line,
    a closing rule. The three-block structure again, at band scale.

    The band is specified by a COUNT, not a pitch, and that is the whole
    point. On a leg the u axis is an unrolled circumference, so u0 and u1
    are the same line - the seam down the back. A band laid out on a fixed
    pitch almost never divides that circumference evenly, so the last stamp
    before the seam collides with the first one after it and one buta comes
    out sliced in half. Dividing the span into `count` equal cells and
    centring a stamp in each puts the seam exactly halfway between two
    stamps, where nothing is. Nothing is ever cut.
    """
    out, fil, det = [], [], []
    h = height
    span = u1 - u0
    cell = span / count

    base = v - h
    out.append(box(u0, base, u1, base + 3.8))                 # the heavy rule
    det.append(box(u0, base + 5.4, u1, base + 7.6))           # its shadow

    # the buta has to fit its cell with room to spare, or the band reads as
    # a row of things touching rather than a row of things placed. The
    # height reserve is arithmetic, not taste: the buta stands 1.71 times
    # its own width, on a 4mm gap above the heavy rule, and has to clear
    # the closing rule at the top. A solid band carries no dotted line, so
    # it needs 13.7mm of furniture instead of 17 and can hold a bigger
    # stamp in a shorter band - which is the whole point of shortening it.
    w = min(cell * 0.66, (h - (13.7 if solid else 17.0)) * 0.585)
    y = base + 7.8
    for k in range(count):
        if solid:
            # one ink, one shape. A silhouette is bolder than a double
            # outline at this size, has nothing in it thinner than the motif
            # itself, and costs one filament instead of three.
            o, f, d = _ok(buta_outline(w)), Polygon(), Polygon()
        else:
            o, f, d = buta_blocks(w, seed=seed * 97 + k, spray=spray,
                                  hashiya=w >= 20.0)
        flip = bool(k % 2)
        for src, dst in ((o, out), (f, fil), (d, det)):
            if src.is_empty:
                continue
            g = affinity.scale(src, -1, 1, origin=(0, 0)) if flip else src
            dst.append(affinity.translate(g, u0 + (k + 0.5) * cell, y))

    if dotted and not solid and h >= 22:
        r = 1.35
        n = count * max(2, int(cell / 13.0))
        dy = v - 6.4
        for k in range(n):
            det.append(Point(u0 + (k + 0.5) * span / n, dy).buffer(r, 10))

    det.append(box(u0, v - 3.9, u1, v - 1.7))                 # closing rule
    return (_ok(clean(unary_union(out))),
            _ok(clean(unary_union(fil))),
            _ok(clean(unary_union([g for g in det if not g.is_empty])),
                area=1.0))


# ------------------------------------------------------------------- field ---

def chownk(u0, u1, v0, v1, ink, pitch=8.4, d=1.8, keep=None):
    """
    The ground dotting - the pass that makes the cloth look bottomless.
    Filtered by vectorised distance rather than by a union-then-difference:
    at this pitch there are several thousand dots and the boolean way costs
    more than the whole rest of the drawing.
    """
    us, vs = [], []
    for i, u in enumerate(np.arange(u0, u1, pitch)):
        off = pitch / 2 if i % 2 else 0.0
        col = np.arange(v0 + off, v1, pitch)
        us.append(np.full(len(col), u)); vs.append(col)
    xy = np.column_stack([np.concatenate(us), np.concatenate(vs)])
    pts = shapely.points(xy)
    shapely.prepare(ink)
    ok = shapely.distance(pts, ink) > (REG + d / 2)
    if keep is not None:
        # WHOLE dots only. Clipping the dotting to the field afterwards is
        # what leaves a fringe of half-dots round the edge of a sole, and a
        # half dot reads as a printing fault rather than as a dot.
        shapely.prepare(keep)
        ok &= shapely.distance(pts, keep.boundary) > (d / 2 + REG)
        ok &= shapely.contains(keep, pts)
    return clean(unary_union([Point(*q).buffer(d / 2, 8) for q in xy[ok]]))


# a pool of carved blocks. A real workshop owns a handful of wooden blocks
# and stamps them over and over; carving a fresh one for every impression
# is both wrong and, at 0.4s a spine, the reason the first field took
# minutes to draw.
_POOL = {}


def pool(w, n=6, spray=True, hashiya=True):
    key = (round(w, 2), n, spray, hashiya)
    if key not in _POOL:
        _POOL[key] = [buta_blocks(w, seed=k, spray=spray, hashiya=hashiya)
                      for k in range(n)]
    return _POOL[key]


def field(pu=50.0, pv=66.0, w_big=31.0, u0=-165.0, u1=165.0,
          v0=-470.0, v1=480.0, seed=1, clip=None, dots=True,
          fillers=True):
    """
    Four blocks at four scales. Each is cut against everything already
    stamped, plus a register gap - which is both how the craft works and
    the only reason a repeat this dense stays legible at a 0.4mm nozzle.
    """
    rng = np.random.default_rng(seed)
    O, F, D = [], [], []
    stamped = []

    def stamp(blocks, u, v, rot=0.0, flip=False):
        gs, out3 = [], []
        for src in blocks:
            if src.is_empty:
                out3.append(None)
                continue
            g = affinity.scale(src, -1, 1, origin=(0, 0)) if flip else src
            if rot:
                g = affinity.rotate(g, rot, origin=(0, 0))
            g = affinity.translate(g, u, v)
            out3.append(g); gs.append(g)
        if not gs:
            return
        whole = unary_union(gs)
        # On a bounded ground - a sole - a stamp that runs off the edge is a
        # fragment, not a motif. A block printer lifts the block rather than
        # printing half of it, so this does too.
        if clip is not None and not clip.contains(whole):
            return
        for g, dst in zip(out3, (O, F, D)):
            if g is not None:
                dst.append(g)
        stamped.append(whole)

    big = pool(w_big)
    small = pool(w_big * 0.55, n=3)

    # 1 - the large buta, half-drop, mirrored column to column
    for i, u in enumerate(np.arange(u0, u1, pu)):
        drop = pv / 2 if i % 2 else 0.0
        for j, v in enumerate(np.arange(v0 + drop, v1, pv)):
            stamp(big[(i * 7 + j) % len(big)], u, v,
                  rot=float(rng.uniform(-7, 7)), flip=bool(i % 2))

    # 2 - the same motif at 55%, upside down, in the gaps
    for i, u in enumerate(np.arange(u0 + pu / 2, u1, pu)):
        drop = pv / 2 if i % 2 else 0.0
        for j, v in enumerate(np.arange(v0 + drop + pv / 2, v1, pv)):
            stamp(small[(i + j) % len(small)], u, v,
                  rot=180.0 + float(rng.uniform(-8, 8)), flip=bool(i % 2))

    guard = clean(unary_union(stamped)) if stamped else Polygon()

    # 3 - a rosette wherever there is still room for one
    gp = []
    if not fillers:
        gp = [(u0 - 1e4, v0 - 1e4)]
    for i, u in enumerate(np.arange(u0, u1, pu / 2)):
        off = pv / 4 if i % 2 else 0.0
        for v in np.arange(v0 + off + pv / 4, v1, pv / 2):
            gp.append((u, v))
    gp = np.array(gp)
    shapely.prepare(guard)
    room = shapely.distance(shapely.points(gp), guard)
    ph = [phool(d, seed=k) for k, d in enumerate((7.5, 9.5, 11.5, 13.0))]
    for (u, v), r in zip(gp, room):
        if r < 7.0:
            continue
        k = int(np.clip((2 * (r - REG) * 0.85 - 7.0) / 2.0, 0, 3))
        a = affinity.translate(ph[k][0], u, v)
        b = affinity.translate(ph[k][1], u, v)
        if clip is not None and not clip.contains(a.union(b)):
            continue
        O.append(a); D.append(b)

    outline = clean(unary_union([g for g in O if not g.is_empty]))
    fill = clean(unary_union([g for g in F if not g.is_empty]))
    detail = clean(unary_union([g for g in D if not g.is_empty]))

    # 4 - the ground dotting, everywhere the other three blocks are not
    # 4 - the ground dotting, everywhere the other three blocks are not
    if dots:
        ink = clean(unary_union([outline, fill, detail]))
        detail = clean(unary_union([detail, chownk(u0, u1, v0, v1, ink,
                                                   keep=clip)]))
    return outline, fill, detail
