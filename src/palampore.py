"""
Paduka - the palampore motif library.

Liberty's Tree of Life is drawn from hand-painted Indian palampore of the
18th and early 19th century: a flowering tree rising from a rocky mound,
sinuous trunk, serrated leaves, big open blossoms, buds, birds. Liberty
reworked the archive into simplified layouts in flat mordant-dye colour -
so these motifs are drawn as bold flat shapes, not as fussy detail, and
they are meant to be read by colour as much as by shadow.

Everything here returns shapely geometry in millimetres.
"""

import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from shapely import affinity


# ---------------------------------------------------------------- helpers ---

def clean(g):
    if g.is_empty:
        return g
    g = g.buffer(0)
    return g


def largest(g):
    if g.is_empty:
        return g
    if isinstance(g, MultiPolygon):
        return max(g.geoms, key=lambda q: q.area)
    return g


def spline(pts, n=260, per=False, smooth=0.0, k=3):
    p = np.asarray(pts, dtype=float)
    if per and np.allclose(p[0], p[-1]):
        p = p[:-1]
    k = min(k, len(p) - 1)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, per=per, k=k)
    u = np.linspace(0, 1, n, endpoint=not per)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def ribbon(path, w0, w1, taper=1.0, cap_end=True, n=240):
    """
    A tapered stroke - the basic unit of a painted stem.

    Palampore branches are brush strokes: thick where they leave the
    trunk, thinning to nothing at the tip. This offsets a centreline by a
    width that falls off along its length.
    """
    c = spline(path, n=n)
    d = np.diff(c, axis=0)
    s = np.concatenate([[0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    s = s / s[-1]

    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
    nrm = np.column_stack([-t[:, 1], t[:, 0]])

    hw = (w0 + (w1 - w0) * s ** taper) / 2.0
    left = c + nrm * hw[:, None]
    right = c - nrm * hw[:, None]

    ring = [left]
    if cap_end and hw[-1] > 0.15:
        a = np.linspace(0, np.pi, 14)
        e, tn = c[-1], t[-1]
        nn = np.array([-tn[1], tn[0]])
        ring.append(np.array([e + nn * hw[-1] * np.cos(x) + tn * hw[-1] * np.sin(x)
                              for x in a]))
    ring.append(right[::-1])
    # round the start
    a = np.linspace(0, np.pi, 14)
    st, tn = c[0], -t[0]
    nn = np.array([-tn[1], tn[0]])
    ring.append(np.array([st + nn * hw[0] * np.cos(x) + tn * hw[0] * np.sin(x)
                          for x in a]))
    return clean(Polygon(np.vstack(ring)))


# ---------------------------------------------------------------- motifs ---

def petal(L=15.0, W=9.0, point=0.86):
    """one painted petal: wide-shouldered, coming to a soft point."""
    c = [(0.00, 0.00), (0.20 * L, 0.30 * W), (0.46 * L, 0.48 * W),
         (0.72 * L, 0.44 * W), (0.90 * L, 0.26 * W), (L * point, 0.02 * W),
         (L, 0.0)]
    up = spline(c, n=90)
    dn = up.copy()[::-1]
    dn[:, 1] *= -1
    return clean(Polygon(np.vstack([up, dn[1:-1]])))


def blossom(d=30.0, petals=6, rot=0.0, gap=1.5, inner=True):
    """
    The palampore blossom: separated painted petals round a lobed heart,
    with a second shorter ring behind them. Returns (petals, heart) so the
    two take different dye colours; the gaps between petals stay as ground
    so the flower reads even before the light does.
    """
    R = d / 2
    out = []
    L = R * 0.80
    W = (2 * np.pi * R * 0.55) / petals - gap
    p0 = petal(L, max(3.4, W))
    for i in range(petals):
        a = np.degrees(rot) + i * 360.0 / petals
        out.append(affinity.translate(
            affinity.rotate(p0, a, origin=(0, 0)), 0, 0))
    if inner and petals >= 5:
        Li, Wi = R * 0.52, max(2.8, W * 0.72)
        pi_ = petal(Li, Wi)
        for i in range(petals):
            a = np.degrees(rot) + (i + 0.5) * 360.0 / petals
            out.append(affinity.rotate(pi_, a, origin=(0, 0)))

    hr = R * 0.30
    ha = np.linspace(0, 2 * np.pi, 260, endpoint=False)
    hl = 1 - 0.16 + 0.16 * np.cos(petals * 2 * (ha - rot))
    heart = clean(Polygon(np.column_stack([hr * hl * np.cos(ha),
                                           hr * hl * np.sin(ha)])))
    ring = clean(unary_union(out).difference(heart.buffer(1.1)))

    # middle rank: shorter petals set between the outer ones, taking the
    # second dye. Palampore flowers are banded, not flat discs.
    mid = []
    Lm, Wm = R * 0.60, max(2.6, W * 0.66)
    pm = petal(Lm, Wm)
    for i in range(petals):
        a = np.degrees(rot) + (i + 0.5) * 360.0 / petals
        mid.append(affinity.rotate(pm, a, origin=(0, 0)))
    mid = clean(unary_union(mid).difference(heart.buffer(0.9)))
    return ring, mid, heart


def leaf(length=26.0, width=11.0, teeth=9, curl=0.30, amp=0.11):
    """serrated palampore leaf with a midrib. Returns (blade, vein)."""
    t = np.linspace(0, 1, 320)
    mid = np.column_stack([length * t,
                           curl * length * (t ** 2) * (1 - 0.45 * t)])
    tg = np.gradient(mid, axis=0)
    tg /= np.linalg.norm(tg, axis=1)[:, None] + 1e-12
    nn = np.column_stack([-tg[:, 1], tg[:, 0]])

    prof = np.sin(np.pi * t ** 0.88) ** 0.72
    saw = (2 / np.pi) * np.arcsin(np.sin(teeth * 2 * np.pi * t))
    hw = (width / 2) * prof * (1 + amp * saw)

    blade = clean(Polygon(np.vstack([mid + nn * hw[:, None],
                                     (mid - nn * hw[:, None])[::-1]])))
    if isinstance(blade, MultiPolygon):
        blade = largest(blade)
    veins = [ribbon(mid[::20], width * 0.155, 0.45, taper=0.75, n=120)]
    # side veins, angled forward off the midrib
    for f in (0.28, 0.46, 0.64, 0.80):
        i = int(f * (len(t) - 1))
        base, nv, tv = mid[i], nn[i], tg[i]
        for sgn in (1, -1):
            tip = base + (nv * sgn * 0.80 + tv * 0.62) * hw[i]
            veins.append(ribbon([base, (base + tip) / 2, tip],
                                width * 0.085, 0.3, taper=0.8, n=40))
    v = clean(unary_union(veins))
    return blade, clean(v.intersection(blade.buffer(-0.42)))


def bud(h=14.0, w=8.0):
    """a closed bud held in pointed sepals."""
    t = np.linspace(0, 1, 160)
    x = (w / 2) * np.sin(np.pi * t) ** 0.70
    y = h * t
    body = clean(Polygon(np.vstack([np.column_stack([x, y]),
                                    np.column_stack([-x[::-1], y[::-1]])])))
    sep = []
    for k, ang in enumerate((-52, 0, 52)):
        s = 1.0 if k == 1 else 0.78
        p = petal(h * 0.46 * s, w * 0.44 * s)
        p = affinity.rotate(p, 90 + ang, origin=(0, 0))
        sep.append(affinity.translate(p, 0, -h * 0.04))
    return body, clean(unary_union(sep))


def _rock(cx, w, h, phase=0.0):
    a = np.linspace(np.pi, 0, 200)
    rr = 1 + 0.085 * np.cos(7 * a + phase)
    x = cx + (w / 2) * np.cos(a) * rr
    y = h * np.sin(a) * rr
    ring = np.vstack([np.column_stack([x, y]),
                      [[cx + w / 2, -2.5], [cx - w / 2, -2.5]]])
    return largest(clean(Polygon(ring)))


def mound(width, height=30.0):
    """
    The rocky mound the tree grows out of - the signature base of every
    palampore. Two ranks of scalloped rocks, the front rank cut out of the
    back one so the overlap reads as depth rather than as a blob.
    """
    back = [_rock(-width * 0.26, width * 0.46, height * 0.86, 0.0),
            _rock(width * 0.04, width * 0.52, height * 1.00, 1.9),
            _rock(width * 0.32, width * 0.44, height * 0.78, 3.4)]
    front = [_rock(-width * 0.40, width * 0.34, height * 0.46, 2.6),
             _rock(-width * 0.12, width * 0.38, height * 0.55, 0.8),
             _rock(width * 0.18, width * 0.36, height * 0.50, 4.2),
             _rock(width * 0.44, width * 0.30, height * 0.40, 1.3)]
    fu = unary_union(front).buffer(1.5)
    carved = [largest(clean(r.difference(fu))) for r in back]
    return clean(unary_union([g for g in carved if not g.is_empty] + front))


def vine_border(sole, inset, amp=3.2, wave=26.0, w=2.6, leaflets=True):
    """
    A running vine round the edge instead of a plain band: the palampore
    border is always a wandering stem, never a rule.
    """
    ring = largest(sole.buffer(-inset)).exterior
    L = ring.length
    n = max(240, int(L / 1.4))
    pts, nrm = [], []
    for i in range(n):
        p = ring.interpolate(L * i / n)
        q = ring.interpolate(L * ((i + 1) % n) / n)
        d = np.array([q.x - p.x, q.y - p.y])
        d /= np.linalg.norm(d) + 1e-12
        pts.append([p.x, p.y])
        nrm.append([-d[1], d[0]])
    pts = np.asarray(pts)
    nrm = np.asarray(nrm)
    ph = np.linspace(0, 2 * np.pi * (L / wave), n, endpoint=False)
    stem = pts + nrm * (amp * np.sin(ph))[:, None]

    band = LineString(np.vstack([stem, stem[:1]])).buffer(w / 2, cap_style=1)

    # leaflets alternating off the wave crests
    if not leaflets:
        return clean(band)
    leaves = []
    for i in range(0, n, max(6, n // 46)):
        s = np.sin(ph[i])
        if abs(s) < 0.72:
            continue
        side = np.sign(s)
        lf, _v = leaf(length=6.4, width=3.4, teeth=3, curl=0.2)
        ang = np.degrees(np.arctan2(nrm[i, 1] * side, nrm[i, 0] * side))
        lf = affinity.rotate(lf, ang, origin=(0, 0))
        lf = affinity.translate(lf, stem[i, 0], stem[i, 1])
        leaves.append(lf)
    return clean(band.union(unary_union(leaves)) if leaves else band)


def tendril(length=22.0, turns=1.35, w0=2.2, w1=0.7, sweep=0.55):
    """a curling tendril, the flourish palampore hangs off a branch end."""
    t = np.linspace(0, 1, 140)
    r = length * (1 - 0.62 * t)
    a = 2 * np.pi * turns * t
    x = length * sweep * t + r * 0.30 * np.cos(a)
    y = r * 0.30 * np.sin(a)
    pts = np.column_stack([x, y])
    return ribbon(pts[::10], w0, w1, taper=0.9, n=160)


def sprig(d=9.0, petals=5, rot=0.0):
    """a tiny strewn flower for the ground between the branches."""
    R = d / 2
    W = (2 * np.pi * R * 0.55) / petals - 0.7
    p0 = petal(R * 0.92, max(2.2, W))
    out = [affinity.rotate(p0, np.degrees(rot) + i * 360.0 / petals,
                           origin=(0, 0)) for i in range(petals)]
    eye = Point(0, 0).buffer(R * 0.26, 32)
    return clean(unary_union(out).difference(eye.buffer(0.7))), clean(eye)


def place(g, x, y, scale=1.0, rot=0.0, flip=False):
    if flip:
        g = affinity.scale(g, xfact=1, yfact=-1, origin=(0, 0))
    if scale != 1.0:
        g = affinity.scale(g, scale, scale, origin=(0, 0))
    if rot:
        g = affinity.rotate(g, rot, origin=(0, 0))
    return affinity.translate(g, x, y)


# ------------------------------------------------- more of the vocabulary ---
# A Liberty floral is never one flower repeated. These are the other
# palampore types the collection leans on: the fan-shaped carnation, the
# split seed pod, the hanging bell, the feathered frond, and the dotted
# rosette that fills the ground between them.

def carnation(w=24.0, h=20.0, petals=7, teeth=4):
    """
    Fan-shaped serrated carnation - the flower palampore draws most often
    after the lotus. Cut into petals by radial slits so it reads as a
    flower and not as a wedge. Returns (fan, calyx).
    """
    span = np.pi * 0.46
    a = np.linspace(-span, span, 420)
    saw = (2 / np.pi) * np.arcsin(np.sin(teeth * petals * (a + span)))
    r = h * (1 + 0.10 * saw)
    pts = np.column_stack([(w / 2) * np.sin(a) / np.sin(span),
                           r * np.cos(a) * 0.92])
    fan = clean(Polygon(np.vstack([pts, [[0.0, h * 0.10]]])))

    # radial slits between the petals
    slits = []
    for i in range(1, petals):
        aa = -span + 2 * span * i / petals
        tip = np.array([(w / 2) * np.sin(aa) / np.sin(span),
                        h * 1.08 * np.cos(aa) * 0.92])
        base = np.array([0.0, h * 0.16])
        slits.append(LineString([base, base + (tip - base) * 1.05])
                     .buffer(0.55, cap_style=2))
    fan = clean(fan.difference(unary_union(slits)))
    if isinstance(fan, MultiPolygon):
        fan = clean(unary_union([g for g in fan.geoms if g.area > 2.0]))

    cal = clean(Polygon([(-w * 0.15, h * 0.20), (w * 0.15, h * 0.20),
                         (w * 0.09, -h * 0.26), (0, -h * 0.36),
                         (-w * 0.09, -h * 0.26)]).buffer(0.6))
    return fan, cal


def pod(h=22.0, w=13.0, seeds=5):
    """a split seed pod with its seeds showing. Returns (husk, seeds)."""
    t = np.linspace(0, 1, 200)
    x = (w / 2) * np.sin(np.pi * t) ** 0.68
    y = h * t
    outer = clean(Polygon(np.vstack([np.column_stack([x, y]),
                                     np.column_stack([-x[::-1], y[::-1]])])))
    # the split: a wedge taken out of the top
    cut = clean(Polygon([(-w * 0.30, h * 0.42), (w * 0.30, h * 0.42),
                         (w * 0.17, h * 1.02), (-w * 0.17, h * 1.02)])
                .buffer(0.4))
    husk = clean(outer.difference(cut))
    sd = []
    for i in range(seeds):
        f = 0.50 + 0.44 * i / max(1, seeds - 1)
        sd.append(Point(0, h * f).buffer(w * 0.135, 26))
    return husk, clean(unary_union(sd).intersection(outer.buffer(-0.8)))


def bell(h=17.0, w=11.0, lobes=3):
    """a hanging bell flower. Returns (bell, stamens)."""
    t = np.linspace(0, 1, 220)
    prof = 0.30 + 0.70 * t ** 1.5
    a = np.linspace(0, np.pi, 90)
    rim = (w / 2) * (1 + 0.13 * np.cos(lobes * 2 * a))
    body = np.vstack([
        np.column_stack([(w / 2) * prof, h * t]),
        np.column_stack([rim * np.cos(a - np.pi), h + 0.10 * h * np.sin(a)])[::-1],
        np.column_stack([-(w / 2) * prof[::-1], h * t[::-1]])])
    g = clean(Polygon(body))
    st = [ribbon([(0, h * 0.98), (0.10 * w, h * 1.16), (0.14 * w, h * 1.30)],
                 1.5, 0.8, n=30),
          ribbon([(0, h * 0.98), (-0.09 * w, h * 1.14), (-0.13 * w, h * 1.26)],
                 1.4, 0.8, n=30)]
    return g, clean(unary_union(st))


def frond(length=30.0, width=13.0, pairs=7, curl=0.22):
    """a feathered frond: rachis with paired leaflets."""
    t = np.linspace(0, 1, 160)
    mid = np.column_stack([length * t, curl * length * t ** 2])
    tg = np.gradient(mid, axis=0)
    tg /= np.linalg.norm(tg, axis=1)[:, None] + 1e-12
    nn = np.column_stack([-tg[:, 1], tg[:, 0]])
    parts = [ribbon(mid[::16], 2.4, 0.7, taper=0.8, n=90)]
    for k in range(pairs):
        f = 0.14 + 0.80 * k / max(1, pairs - 1)
        i = int(f * (len(t) - 1))
        L = (width / 2) * (1 - 0.55 * f) * 1.9
        for sgn in (1, -1):
            lf, _v = leaf(L, L * 0.44, teeth=3, curl=0.18)
            ang = np.degrees(np.arctan2(nn[i, 1] * sgn + tg[i, 1] * 0.55,
                                        nn[i, 0] * sgn + tg[i, 0] * 0.55))
            lf = affinity.rotate(lf, ang, origin=(0, 0))
            parts.append(affinity.translate(lf, mid[i, 0], mid[i, 1]))
    return clean(unary_union(parts))


def dotted_rosette(d=11.0, dots=7):
    """small flower: a disc heart ringed by dots - pure ground filler."""
    r = d / 2
    heart = Point(0, 0).buffer(r * 0.34, 30)
    ring = []
    for i in range(dots):
        a = 2 * np.pi * i / dots
        ring.append(Point(r * 0.70 * np.cos(a), r * 0.70 * np.sin(a))
                    .buffer(r * 0.24, 20))
    return clean(unary_union(ring)), clean(heart)


# --------------------------------------------- piercing and spear leaves ---
# In the Liberty prints the motifs are not flat silhouettes: they are shot
# through with tiny reserved dots and fine hatching, so the pale ground
# reads back through the colour. In relief the same trick works by
# piercing the raised shape - the ground shows in the gaps.

def pierce_dots(g, spacing=3.5, r=0.78, margin=1.5, stagger=True):
    """punch a reserved dot field out of a raised shape."""
    if g.is_empty:
        return g
    inner = g.buffer(-margin)
    if inner.is_empty:
        return g
    b = inner.bounds
    dots, row = [], 0
    y = b[1]
    while y <= b[3]:
        off = (spacing / 2 if (stagger and row % 2) else 0.0)
        x = b[0] + off
        while x <= b[2]:
            p = Point(x, y)
            if inner.contains(p):
                dots.append(p.buffer(r, 12))
            x += spacing
        y += spacing * 0.87
        row += 1
    return clean(g.difference(unary_union(dots))) if dots else g


def pierce_hatch(g, spacing=2.6, angle=35.0, w=0.55, margin=1.1):
    """rule fine reserved lines across a raised shape."""
    if g.is_empty:
        return g
    inner = g.buffer(-margin)
    if inner.is_empty:
        return g
    b = inner.bounds
    d = max(b[2] - b[0], b[3] - b[1]) * 1.6
    cxm, cym = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    a = np.radians(angle)
    ux, uy = np.cos(a), np.sin(a)
    lines = []
    k = -int(d / spacing) - 1
    while k * spacing <= d:
        off = k * spacing
        px, py = cxm - uy * off, cym + ux * off
        lines.append(LineString([(px - ux * d, py - uy * d),
                                 (px + ux * d, py + uy * d)])
                     .buffer(w / 2, cap_style=2))
        k += 1
    cut = unary_union(lines).intersection(inner)
    return clean(g.difference(cut))


def spear_leaf(length=38.0, width=9.0, teeth=15, curl=0.16):
    """
    The long narrow spear leaf that runs through both colourways - much
    more lance than the broad serrated leaf, and the thing that gives the
    print its spiky rhythm.
    """
    t = np.linspace(0, 1, 340)
    mid = np.column_stack([length * t, curl * length * (t ** 1.7)])
    tg = np.gradient(mid, axis=0)
    tg /= np.linalg.norm(tg, axis=1)[:, None] + 1e-12
    nn = np.column_stack([-tg[:, 1], tg[:, 0]])
    prof = np.sin(np.pi * t ** 0.62) ** 1.15
    saw = (2 / np.pi) * np.arcsin(np.sin(teeth * 2 * np.pi * t))
    hw = (width / 2) * prof * (1 + 0.10 * saw)
    blade = clean(Polygon(np.vstack([mid + nn * hw[:, None],
                                     (mid - nn * hw[:, None])[::-1]])))
    if isinstance(blade, MultiPolygon):
        blade = largest(blade)
    rib = ribbon(mid[::24], width * 0.15, 0.4, taper=0.7, n=110)
    return blade, clean(rib.intersection(blade.buffer(-0.4)))


def pomegranate(d=28.0, crown=0.26):
    """
    The seeded fruit both colourways are built around: a round body with a
    little crown, its belly filled with a lattice of seeds.
    Returns (husk, seeds).
    """
    a = np.linspace(0, 2 * np.pi, 320, endpoint=False)
    r = (d / 2) * (1 - 0.06 * np.cos(3 * a) + 0.05 * np.sin(a))
    body = clean(Polygon(np.column_stack([r * np.cos(a), r * np.sin(a)])))
    cr = []
    for k, ang in enumerate((-34, -11, 11, 34)):
        p = petal(d * crown * (0.8 if k in (0, 3) else 1.0), d * 0.13)
        cr.append(affinity.rotate(p, 90 + ang, origin=(0, 0)))
    crown_g = affinity.translate(unary_union(cr), 0, d * 0.42)
    husk = clean(unary_union([body, crown_g]))
    # seeds: a staggered lattice in the lower belly
    seeds = []
    inner = body.buffer(-d * 0.13)
    b = inner.bounds
    sp = d * 0.155
    row = 0
    y = b[1]
    while y <= b[3]:
        x = b[0] + (sp / 2 if row % 2 else 0)
        while x <= b[2]:
            q = Point(x, y)
            if inner.contains(q):
                seeds.append(q.buffer(sp * 0.30, 16))
            x += sp
        y += sp * 0.9
        row += 1
    return husk, clean(unary_union(seeds)) if seeds else (husk, Polygon())
