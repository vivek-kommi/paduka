"""
Paduka - the Tree of Life composition, drawn as a trail.

A palampore tree: it grows out of the heel, the trunk sweeps up the arch
(passing behind the adapter boss), and it comes into flower over the ball
and toe - the part of the foot that leads. Liberty's own note on the
collection is that Tree of Life textiles "symbolise the personal journey
through nature's life cycle", which is not a bad thing for a prosthesis
to say.

Liberty describe their own rework of the archive as taking a design that
"has historically appeared as a best-selling scarf design" and putting it
into "a simpler layout, to enhance the striking floral motifs", printed
"on linen in flat colour" in the "bold colouration of mordant dyes and
natural pigments". That is the brief this file follows, and it is a
trail rather than a tree: one undulating stem the length of the plate,
fourteen large full-face rosettes set alternately above and below it,
big serrated leaves between them, and a plain running vine at the edge.
About 55% of the plate is inked, and the rest is left as linen.

Two things are deliberately absent. There is no reserved hatching or
dotting - the shapes are flat colour, which is what the fabrics are. And
the blossom is two colours rather than three: at 20-35mm with a printable
ground line between dyes, a third rank of petals cannot survive, and the
detail of it is explained where blossom_parts is defined.

Everything here is sized for a 0.4mm nozzle. The constants that do that
work are MIN_W, MIN_AREA and OUTLINE, and printable() is what enforces
them - a shape the nozzle cannot lay down cleanly is removed here rather
than left for the slicer to turn into gap-fill and stringing.

Drawn in three flat colours the way Liberty reworked the archive: undyed
linen ground, jade, lapis.
"""

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, box, LineString
from shapely.ops import unary_union
from shapely import affinity

import palampore as PM
from palampore import clean, largest, ribbon, place, spline
from paduka_pattern import sole_outline
import paduka_pattern as PP

# --- dimensions -------------------------------------------------------------
PLATE_T   = 14.0
H_STEM    = 1.15      # stems, leaves and buds sit this proud
H_FLOWER  = 1.85      # blossoms sit proud of the foliage
BEVEL     = 0.28
PED_H     = 26.0
BOSS_D    = 30.0
BOSS_H    = 8.0
BOLT_PITCH = 25.0
BOLT_D    = 6.6
BOSS_X    = 112.0

# --- modular attachment (ISO 10328 practice) --------------------------------
PAD_D      = 48.0     # flat machined landing for the adapter
PAD_H      = 2.70     # MUST stand proud of the tallest ornament, or the
                      # adapter seats on the pattern instead of on the
                      # landing and sits crooked. H_FLOWER + 0.85.
LANDING_CLEAR = 3.0   # ring of bare ground kept round the landing for the
                      # pattern and the inner vine. The outer band is held
                      # to a tighter 0.8mm because at the waist the plate
                      # is not wide enough for both, and the band hugs the
                      # extreme edge where nothing can reach it anyway.
SPIGOT_D   = 48.0     # material added underneath for thread depth
SPIGOT_H   = 10.0
INSERT_D   = 8.2      # M6 brass heat-set insert
INSERT_DEP = 13.0
CENTRE_D   = 10.5     # M10 clearance for single-bolt foot adapters
CENTRE_CB  = 20.0     # counterbore underneath for its nut and washer
CENTRE_CBD = 8.0

BORDER_IN = 3.4      # the outer band runs along the actual edge
BORDER_AMP  = 1.7    # how far the inner vine wanders off its ring
BORDER_WAVE = 24.0   # and how long one wave of it is
BORDER_LEAF = 5.6    # length of the leaves growing off the vine
FIELD_IN  = 5.6      # and the pattern covers the whole plate

JADE   = "jade"
LAPIS  = "lapis"
LEAF   = "leaf"       # jade at stem height, but drawn under the trail
STEMS  = "stems"      # jade at stem height, merged into one body
# each Liberty colourway is monochrome on a pale ground, so there is no
# third hue: only the two dye tones and the linen showing through

OUTLINE = 1.00       # ground line left between shapes of different dye

# --- sized for a 0.4mm nozzle -----------------------------------------------
MIN_W      = 1.20    # minimum width of any raised feature (mm)
MIN_AREA   = 9.0     # minimum island footprint (mm2)
STEM_REACH = 38.0    # a filled motif further than this from the trail is
                     # left out rather than left floating
TARGET_COV = 0.90    # stop filling at this fraction of the field.
                     # Measured against the RAW shapes, which overlap each
                     # other and have not yet had the ground line cut out
                     # of them, so it reads higher than the finished
                     # coverage - what really stops the filler is running
                     # out of gaps big enough to hold anything.


def pedestal_footprints(sole):
    def carve(x0, x1):
        g = largest(sole.buffer(-19.0).intersection(box(x0, -100, x1, 100)))
        return clean(g.buffer(-8.0).buffer(8.0)) if not g.is_empty else g
    return carve(8.0, 62.0), carve(160.0, 218.0)


def bolt_holes(cy):
    """the four M6 positions on the 25mm square."""
    h = BOLT_PITCH / 2
    return [Point(BOSS_X + dx, cy + dy) for dx in (-h, h) for dy in (-h, h)]


def adapter_pad(cy):
    """the flat landing the adapter seats on."""
    return Point(BOSS_X, cy).buffer(PAD_D / 2, 96)


# --- the trail --------------------------------------------------------------
# Authored running along +x, heel to toe, and given real amplitude so it
# wanders the plate rather than ruling a line down it. It passes behind
# the adapter landing, so it is cut there and picks up on the far side,
# which reads as depth rather than as a break.

TRUNK = [(6, 5), (24, -3), (44, 3), (66, -6), (88, 1), (110, -6),
         (134, -1), (156, -8), (178, 0), (200, -7), (222, 1), (244, 6)]
TRUNK_W0, TRUNK_W1 = 5.8, 2.0

# The rosettes. Placed as coordinates rather than derived from the trail:
# a trail hangs its flowers off the stem, but this plate is 96mm wide
# with a 48mm bare landing in the middle of it, and fitting fifteen
# 30mm flowers to a stem that has to detour round that leaves half of
# them shrunk to buttons. So the flowers are placed where the plate has
# room for them, and the trail is then routed to thread through them -
# which is the same relationship, drawn the other way round. Each one is
# checked against the plate edge and the landing at build time and shrunk
# if it fouls either.
#   x, y, diameter, petals, rotation
BLOSSOMS = [
    ( 22, -14, 32, 8, 0.40), ( 30,  17, 27, 7, 1.10),
    ( 56, -18, 31, 8, 0.15), ( 58,  17, 27, 7, 0.90),
    ( 82, -18, 25, 7, 1.50), ( 80,  13, 24, 6, 0.55),
    (146, -25, 31, 8, 0.30), (146,  12, 27, 7, 1.00),
    (182, -30, 38, 9, 0.60), (180,  20, 33, 8, 0.20),
    (214, -24, 33, 8, 1.20), (215,  19, 30, 8, 0.70),
    (198,   1, 23, 6, 0.35), (240, -10, 26, 7, 0.45),
    (242,  16, 22, 6, 1.30),
    # a second rank of smaller rosettes, set between the large ones so
    # the trail reads as a sequence of flowers rather than a row of them
    ( 44,   0, 19, 6, 0.60), ( 70,   2, 18, 6, 1.40),
    (164,  -8, 21, 7, 0.20), (198, -16, 20, 6, 0.80),
    (230,   6, 20, 6, 1.50), (252,   2, 17, 5, 0.90),
]


# --- printability -----------------------------------------------------------

def _parts(g):
    if g.is_empty:
        return []
    if isinstance(g, MultiPolygon):
        return [p for p in g.geoms if p.geom_type == "Polygon" and p.area > 0]
    return [g] if g.geom_type == "Polygon" else []


def printable(g, w=None, a=None):
    """
    Open the shape by half the minimum feature width and throw away what
    does not survive. A raised feature thinner than w is not something a
    0.4mm nozzle draws - it is gap-fill, or a whisker dragged across the
    plate - and an island smaller than a is a travel move waiting to
    string. Both are removed here, while the geometry is still geometry.
    Corners come back slightly rounded, which is what a printed edge
    looks like anyway.
    """
    if g.is_empty:
        return g
    w = MIN_W if w is None else w
    a = MIN_AREA if a is None else a
    o = clean(g.buffer(-w / 2, join_style=1).buffer(w / 2, join_style=1))
    o = clean(o.intersection(g))          # never grow past the original
    keep = [p for p in _parts(o) if p.area >= a]
    return clean(unary_union(keep)) if keep else Polygon()


# --- why the blossom is two colours, not three ------------------------------
# The palampore blossom is properly banded: an outer rank of petals, a
# shorter rank behind it in the second dye, and a lobed heart. Drawn at
# 20-35mm with a 1mm ground line between the dyes, that does not survive
# contact with a 0.4mm nozzle. The middle rank sits between the outer
# petals, so it needs its ground line on both sides, and at this scale
# the two gaps are wider than the space between the outer petals - the
# outer rank gets eaten back to its tips and a 26mm flower resolves into
# eighteen islands of under 3mm2. Measured, not guessed. So the rank
# goes: petals in one dye, heart in the other, alternating flower to
# flower so the plate still reads in two colours. Fewer parts, fatter
# parts, and it still reads as a flower from standing height, which is
# the only distance that matters.

def blossom_parts(d, petals, rot, swap=False):
    ring, _mid, heart = PM.blossom(d, petals=petals, rot=rot,
                                   gap=2.0, inner=False)
    a, b = (JADE, LAPIS) if swap else (LAPIS, JADE)
    return [(ring, a, H_FLOWER), (heart, b, H_FLOWER)]


def _fill_motif(room, rot, rng):
    """
    The largest motif that fits in `room` mm of clear ground, or None.
    Sizes are chosen so the motif is always smaller than the gap it was
    picked for - a motif that overflows its gap gets cut back by whatever
    it collides with, and what is left is slivers.
    """
    if room >= 8.4:
        d = float(np.clip(2 * room * 0.90, 15.0, 26.0))
        return blossom_parts(d, int(rng.integers(6, 8)), float(rng.random()),
                             swap=bool(rng.integers(0, 2))), d / 2

    if room >= 5.2:
        d = float(np.clip(2 * room * 0.94, 9.6, 15.0))
        ring, eye = PM.sprig(d, petals=int(rng.integers(5, 7)),
                             rot=float(rng.random()))
        return [(ring, LAPIS, H_FLOWER), (eye, JADE, H_FLOWER)], d / 2

    return None


def palampore_border(sole):
    """
    A two-run border rather than a single rule: a plain band along the
    edge in one dye, and just inside it a wavy vine in the other with
    small leaves growing off the crests.

    The leaves are drawn touching the vine on purpose. A border of
    separate beads would look right and print badly - a hundred little
    islands is a hundred travel moves and a hundred chances to string -
    whereas leaves joined to their stem come out as ONE island for the
    whole run. Two islands for the entire border, and it reads richer
    than the single line it replaces.
    """
    def ring_pts(inset, n_min=520):
        r = largest(sole.buffer(-inset)).exterior
        L = r.length
        n = max(n_min, int(L / 1.2))
        pts, nrm = [], []
        for i in range(n):
            p = r.interpolate(L * i / n)
            q = r.interpolate(L * ((i + 1) % n) / n)
            d = np.array([q.x - p.x, q.y - p.y])
            d /= np.linalg.norm(d) + 1e-12
            pts.append([p.x, p.y])
            nrm.append([-d[1], d[0]])
        return np.asarray(pts), np.asarray(nrm), L

    # the outer band
    pts, _n, _L = ring_pts(BORDER_IN)
    band = LineString(np.vstack([pts, pts[:1]])).buffer(1.45, cap_style=1)

    # the inner vine, and the leaves off it
    pts, nrm, L = ring_pts(BORDER_IN + 5.4)
    n = len(pts)
    ph = np.linspace(0, 2 * np.pi * (L / BORDER_WAVE), n, endpoint=False)
    stem = pts + nrm * (BORDER_AMP * np.sin(ph))[:, None]
    vine = [LineString(np.vstack([stem, stem[:1]])).buffer(0.95, cap_style=1)]

    # leaves on the inward crests only. Growing them both ways looks
    # right on paper but the outward ones touch the band and chop it into
    # dashes, and a dashed band is a dozen extra islands as well as being
    # wrong.
    step = max(8, n // 62)
    for i in range(0, n, step):
        s_ = np.sin(ph[i])
        if s_ > -0.55:
            continue
        side = -1.0
        lf, _v = PM.leaf(BORDER_LEAF, BORDER_LEAF * 0.60, teeth=3,
                         curl=0.22)
        ang = np.degrees(np.arctan2(nrm[i, 1] * side, nrm[i, 0] * side))
        lf = affinity.rotate(lf, ang, origin=(0, 0))
        # pulled back onto the stem so the leaf and the vine are one body
        vine.append(affinity.translate(lf, stem[i, 0] - nrm[i, 0] * side * 0.8,
                                       stem[i, 1] - nrm[i, 1] * side * 0.8))
    return clean(band), clean(unary_union(vine))


def _fit_blossoms(field, pad):
    """Keep each rosette where it is; shrink it until it clears the plate
    edge and the adapter landing. Below 16mm it is dropped rather than
    printed as a button."""
    out, done_g = [], []
    for x, y, d, petals, rot in BLOSSOMS:
        placed = False
        for d_try in np.arange(d, 15.0, -1.5):
            base = unary_union([q for q, _c, _h in
                                blossom_parts(d_try, petals, rot)])
            # try where it was asked for, then pulled in towards the
            # centreline - the border takes a 10mm ring off the plate and
            # a flower is better moved than shrunk
            for pull in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
                yy = y - np.sign(y) * pull
                g = place(base, x, yy)
                # test the flower, not its bounding circle: a rosette is
                # mostly gaps, and the circle fails long before the
                # petals do
                if (not field.contains(g.buffer(0.4)) or g.intersects(pad)
                        or any(g.intersects(q) for q in done_g)):
                    continue
                out.append((float(x), float(yy), float(d_try),
                            int(petals), float(rot)))
                done_g.append(g.buffer(OUTLINE))
                placed = True
                break
            if placed:
                break
    return out


def _trunk_curve(n=400):
    return spline(TRUNK, n=n)


def _branch_to(cx, cy, curve, back=18.0):
    """A short stalk from the trail out to a flower."""
    d = np.hypot(curve[:, 0] - cx, curve[:, 1] - cy)
    i = int(np.argmin(d))
    step = np.hypot(*(curve[1] - curve[0]))
    j = max(0, i - int(back / max(step, 1e-6)))
    a = curve[j]
    tip = np.array([cx, cy], dtype=float)
    v = tip - a
    L = float(np.hypot(*v))
    if L < 5.0:
        return None
    u = v / L
    perp = np.array([-u[1], u[0]])
    bow = 0.16 * L * (1 if cy >= a[1] else -1)
    m1 = a + u * (L * 0.36) + perp * bow * 0.55
    m2 = a + u * (L * 0.72) + perp * bow * 0.85
    return [tuple(a), tuple(m1), tuple(m2), tuple(tip * 0.98 + a * 0.02)], L


def build_tree():
    sole = sole_outline()
    field = largest(sole.buffer(-FIELD_IN))
    cy0 = PP.centre_at(sole, BOSS_X)
    pad = Point(BOSS_X, cy0).buffer(PAD_D / 2 + LANDING_CLEAR, 96)
    field = clean(field.difference(pad))
    cy = cy0

    # The border is drawn first and then subtracted from the field, so
    # nothing in the pattern can run into it and chop it into dashes.
    band, vine = palampore_border(sole)
    landing = Point(BOSS_X, cy0).buffer(PAD_D / 2 + 0.8, 96)
    band = clean(band.intersection(sole.buffer(-0.7)).difference(landing))
    vine = clean(vine.intersection(sole.buffer(-0.7)).difference(pad))
    field = clean(field.difference(
        unary_union([band, vine]).buffer(OUTLINE * 1.2)))

    blossoms = _fit_blossoms(field, pad)
    stack, holes, stems, leaves = [], [], [], []
    rng = np.random.default_rng(7)

    def add(g, colour, h):
        """
        Queue a shape. Order is paint order: whatever is added later sits
        on top, and is cut out of everything beneath it with a thin line
        of undyed ground left showing - which is what a painted outline
        does on a palampore, and what keeps the shapes legible once they
        are all the same height in relief.

        STEMS and LEAF are jade at stem height like everything else, but
        they are held back: the stems all merge into one connected body
        before slicing, and the trail is drawn over the leaves rather
        than under them.
        """
        if g.is_empty:
            return
        g = clean(g.intersection(field))
        if g.is_empty or g.area <= 1.0:
            return
        if colour == STEMS:
            stems.append(g)
        elif colour == LEAF:
            leaves.append((g, JADE, h))
        else:
            stack.append((g, colour, h))

    # --- the trail and its stalks ----------------------------------------
    curve = _trunk_curve()
    branches = []
    for x, y, d, _n, _r in blossoms:
        made = _branch_to(x, y, curve, back=10.0 + 0.14 * d)
        if made is None:
            continue
        path, L = made
        branches.append((path, L))
        add(ribbon(path, np.clip(0.016 * L + 2.4, 2.8, 4.2), 1.6, taper=0.85),
            STEMS, H_STEM)
    add(ribbon(TRUNK, TRUNK_W0, TRUNK_W1, taper=0.9), STEMS, H_STEM)

    # --- leaves ------------------------------------------------------------
    # Big serrated blades, solid: the reserved midrib and side veins of
    # the first version were hatching by another name and were the first
    # thing to turn to mush at this scale.
    for path, L in branches:
        c = spline(path, n=200)
        tg = np.gradient(c, axis=0)
        tg /= np.linalg.norm(tg, axis=1)[:, None] + 1e-12
        for k in range(3):
            i = int((0.28 + 0.26 * k) * (len(c) - 1))
            side = 1.0 if k % 2 == 0 else -1.0
            ang = np.degrees(np.arctan2(tg[i, 1], tg[i, 0]))
            ang += side * (60.0 + rng.uniform(-7, 7))
            Ln = float(np.clip(0.85 * L, 18.0, 28.0)) * (1 - 0.10 * k)
            bl, _v = PM.leaf(Ln, Ln * 0.46, teeth=max(5, int(Ln / 3.4)),
                             curl=0.30)
            g = place(bl, c[i, 0], c[i, 1], rot=ang, flip=side < 0)
            if g.intersection(field).area < 0.72 * g.area:
                continue      # a leaf half off the plate is just a smear
            add(g, LEAF, H_STEM)

    tc = spline(TRUNK, n=200)
    ttg = np.gradient(tc, axis=0)
    ttg /= np.linalg.norm(ttg, axis=1)[:, None] + 1e-12
    for k, f in enumerate((0.035, 0.085, 0.135, 0.185, 0.235, 0.285,
                           0.335, 0.545, 0.595, 0.645, 0.695, 0.745,
                           0.795, 0.845, 0.895, 0.940, 0.975)):
        i = int(f * (len(tc) - 1))
        side = 1.0 if k % 2 else -1.0
        ang = np.degrees(np.arctan2(ttg[i, 1], ttg[i, 0])) + side * 58.0
        Ln = 28.0 - 0.8 * abs(k - 8.0)
        bl, _v = PM.leaf(Ln, Ln * 0.46, teeth=max(5, int(Ln / 3.4)),
                         curl=0.30)
        g = place(bl, tc[i, 0], tc[i, 1], rot=ang, flip=side < 0)
        if g.intersection(field).area < 0.72 * g.area:
            continue
        add(g, LEAF, H_STEM)

    # --- the rosettes ------------------------------------------------------
    for k, (x, y, d, n, rot) in enumerate(blossoms):
        for g, c_, h in blossom_parts(d, n, rot, swap=bool(k % 3 == 1)):
            add(place(g, x, y), c_, h)

    # --- border ------------------------------------------------------------
    stack.append((band, LAPIS, H_STEM))
    stack.append((vine, JADE, H_STEM))

    # --- filling what is left ----------------------------------------------
    # Each filled motif goes into the largest gap that will actually hold
    # it, at the largest size that fits, and is then tied back to the
    # nearest stem with a short branch - so the ground fills with growth
    # rather than confetti, and, just as importantly, the filler is not a
    # scatter of isolated islands for the nozzle to travel between.
    stem_net = LineString(tc)
    for path, _L in branches:
        stem_net = unary_union([stem_net, LineString(spline(path, n=40))])

    covered = clean(unary_union([g for g, _c, _h in stack + leaves] + stems))
    b = field.bounds
    gx, gy = np.meshgrid(np.arange(b[0] + 2, b[2], 2.4),
                         np.arange(b[1] + 2, b[3], 2.4))
    pts = shapely.points(np.column_stack([gx.ravel(), gy.ravel()]))
    keep = shapely.contains(field.buffer(-1.0), pts)
    pts = pts[keep]
    xy = shapely.get_coordinates(pts)
    d_ink = np.minimum(shapely.distance(pts, covered),
                       shapely.distance(pts, field.boundary))
    d_stem = shapely.distance(pts, stem_net)

    ink = covered.area
    n_fill = 0
    for _ in range(400):
        if ink / field.area >= TARGET_COV:
            break
        room = d_ink - OUTLINE
        room[d_stem > STEM_REACH] = 0.0
        i = int(np.argmax(room))
        if room[i] < 3.6:
            break
        x, y, r = float(xy[i, 0]), float(xy[i, 1]), float(room[i])
        near = stem_net.interpolate(stem_net.project(Point(x, y)))
        rot = float(np.degrees(np.arctan2(y - near.y, x - near.x)))

        made = _fill_motif(r, rot, rng)
        if made is None:
            d_ink[i] = 0.0
            continue
        parts_, need = made
        if need > r + 0.4:
            d_ink[i] = 0.0
            continue
        placed = [(place(g, x, y, rot=rot - 90.0), c_, h)
                  for g, c_, h in parts_]
        whole = unary_union([g for g, _c, _h in placed])
        if (not field.contains(whole.buffer(0.8))
                or whole.distance(covered) < OUTLINE * 0.85):
            d_ink[i] = 0.0
            continue

        gap = Point(x, y).distance(near)
        if need * 0.9 < gap < need + 11.0:
            add(ribbon([(near.x, near.y),
                        ((near.x + x) / 2, (near.y + y) / 2), (x, y)],
                       3.0, 1.8, taper=0.9, n=60), STEMS, H_STEM)
        for g, c_, h in placed:
            add(g, c_, h)
        n_fill += 1
        ink += whole.difference(covered).area
        covered = clean(unary_union([covered, whole]))
        d_ink = np.minimum(d_ink, shapely.distance(pts, whole))

    # --- the modular attachment --------------------------------------------
    # A modern foot bolts to a pyramid adapter, so it needs a flat
    # machined landing, a standard hole pattern, and enough material
    # under it to hold the threads. The ornament stops short of it.
    holes.append(Point(BOSS_X, cy).buffer(CENTRE_D / 2, 64))
    ped_h, ped_f = pedestal_footprints(sole)
    core = unary_union([p.buffer(-5.0) for p in (ped_h, ped_f)])
    any_p = unary_union([ped_h, ped_f])
    holes = [h for h in holes
             if (not h.intersects(any_p)) or core.contains(h.buffer(1.5))]

    # --- resolve the paint order -------------------------------------------
    # Leaves first, then the trail over them so it reads as one line, then
    # the flowers. The stems are merged into a single body before they go
    # down: the trail and every stalk off it as one connected mass rather
    # than ninety overlapping ribbons, which is where island count - and
    # so travel moves, and so stringing - comes from. Every shape then
    # goes through printable() before it is kept.
    order = list(leaves)
    if stems:
        order.append((clean(unary_union(stems)), JADE, H_STEM))
    order += stack

    # The ground line is only cut between shapes of DIFFERENT dye. Two
    # jade leaves that touch just become one jade leaf-shaped body: no
    # colour boundary, nothing to bleed, and one island instead of two.
    # Reserving the gap only where the colours actually meet is what lets
    # the drawing pack this tightly without going below what the nozzle
    # can lay down.
    resolved = []
    above = {JADE: Polygon(), LAPIS: Polygon()}
    for g, colour, h in reversed(order):
        other = unary_union([v for k, v in above.items() if k != colour])
        cut = g
        if not other.is_empty:
            cut = clean(cut.difference(other.buffer(OUTLINE)))
        if not above[colour].is_empty:
            cut = clean(cut.difference(above[colour]))
        cut = printable(cut)
        if not cut.is_empty and cut.area > 0.8:
            resolved.append((cut, colour, h))
        above[colour] = clean(unary_union([above[colour], g]))

    jade = [(g, h) for g, c_, h in resolved if c_ == JADE]
    lapis = [(g, h) for g, c_, h in resolved if c_ == LAPIS]

    def merge(layers):
        out = {}
        for g, h in layers:
            out.setdefault(round(h, 3), []).append(g)
        return {h: clean(unary_union(v)) for h, v in out.items()}

    return dict(sole=sole, field=field, jade=merge(jade), lapis=merge(lapis),
                holes=clean(unary_union(holes)), ped_heel=ped_h,
                ped_fore=ped_f, bolt_cy=cy, n_hole=len(holes),
                n_blossom=len(blossoms), n_fill=n_fill)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from preview2d import draw

    T = build_tree()
    fig, ax = plt.subplots(figsize=(17, 7))
    draw(ax, T["sole"], fc="#E7DEC8", ec="none")
    for g in T["jade"].values():
        draw(ax, g, fc="#1E6B5E", ec="none")
    for g in T["lapis"].values():
        draw(ax, g, fc="#26407A", ec="none")
    draw(ax, T["holes"], fc="#1a1713", ec="none")

    ax.set_xlim(-6, 266); ax.set_ylim(-56, 52)
    ax.set_aspect("equal"); ax.axis("off")
    plt.savefig("_tree.png", dpi=130, bbox_inches="tight")
    print("holes", T["n_hole"], "| rosettes", T["n_blossom"],
          "| filled motifs", T["n_fill"])
