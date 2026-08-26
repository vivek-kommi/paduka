"""
Paduka - the foot, built as anatomy rather than as an extruded outline.

The earlier attempts lofted horizontal slices of the sandal outline and
eroded them as they rose. That gives a wedge - a rock, not a foot - because
a foot's defining features are all in its CROSS SECTION, and a stack of
horizontal slices cannot express them.

So this builds it the other way up: a run of cross-sections taken
perpendicular to the length, at stations from the heel to the toe-break,
lofted along x. Each station carries four numbers - centreline, half-width,
ground clearance and dorsum height - and between them they carry the four
things that make a foot read as a foot:

  the arch      the medial border lifts clear of the ground between heel
                and ball, so the print is a heel pad, a lateral band and a
                ball, not a solid slab
  the ball      widest station in the foot, at about two thirds of length
  the dorsum    a domed instep falling from ankle to toe-break, which is
                what catches the light
  the heel      narrower than the ball and rounded under, not a cylinder

The toes are separate bodies, not part of the loft: five tapered capsules,
splayed, each resting on the ground at its own radius, big toe carrying a
lifted nail bed. And the cleft between the great toe and the second is
where the paduka's post comes up - so the anatomy and the fixing are the
same feature, which is the whole argument of the project in one detail.

The ankle is a short column with two malleoli, the medial set higher and
further forward than the lateral, as they are on a person.
"""

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as rot

from paduka_leg import loft

# --- the stations -----------------------------------------------------------
# x, centreline y, half-width, ground clearance, dorsum height
#
# Ground clearance is the arch. It is zero under the heel and under the
# ball, and rises to 7mm through the waist - which is the single number
# that stops this reading as a slipper.
STATIONS = [
    (  0.5,  -1.0,  8.0, 14.0, 24.0),
    (  3.0,  -1.0, 15.0,  8.0, 31.0),
    (  8.0,  -1.0, 21.0,  2.4, 44.0),
    ( 13.0,  -1.2, 24.0,  0.6, 51.0),
    ( 20.0,  -1.6, 27.5,  0.0, 59.0),
    ( 28.0,  -2.0, 29.0,  0.0, 65.0),
    ( 38.0,  -2.5, 30.0,  0.5, 69.0),
    ( 50.0,  -3.0, 30.0,  1.8, 70.0),
    ( 62.0,  -3.4, 29.8,  3.2, 68.0),
    ( 74.0,  -3.8, 29.2,  4.4, 64.0),
    ( 88.0,  -4.0, 29.0,  5.2, 58.5),
    (102.0,  -3.8, 29.8,  5.5, 53.0),
    (116.0,  -3.2, 31.0,  5.2, 48.0),
    (130.0,  -2.2, 33.0,  4.2, 44.0),
    (142.0,  -1.0, 35.0,  3.0, 41.5),
    (152.0,   0.4, 37.0,  1.9, 39.0),
    (161.0,   1.6, 38.8,  1.0, 36.8),
    (170.0,   2.6, 40.0,  0.3, 34.5),
    (178.0,   3.2, 40.4,  0.0, 32.3),
    (187.0,   3.5, 39.8,  0.0, 29.6),
    (196.0,   3.5, 38.4,  0.0, 26.6),
    (204.0,   3.2, 36.6,  0.0, 23.6),
    (210.0,   2.8, 34.8,  0.0, 21.0),
    (214.0,   2.5, 33.0,  0.0, 19.0),
]


TOE_BREAK = 214.0

# name, y, length, radius, splay degrees, lift
TOES = [
    ("great",  20.0, 32.0, 10.6,  -5.0, 0.6),
    ("second",  6.5, 30.0,  8.0,  -1.5, 0.4),
    ("third",  -5.5, 27.0,  7.4,   2.0, 0.3),
    ("fourth",-16.0, 23.0,  6.8,   6.0, 0.2),
    ("little",-25.5, 18.0,  6.0,  11.0, 0.1),
]

ANKLE_X = 40.0


def _section(x, cy, w, z0, z1, n=96, squareness=2.4, flat=1.45, dome=0.86):
    """
    One cross-section, in the plane perpendicular to the length.

    The upper half is domed and the lower half is flattened - a foot in
    section is not an ellipse, it is a dome sitting on a sole. `flat`
    raises the lower half toward a straight line; `dome` softens the top.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    y = w * np.sign(c) * np.abs(c) ** (2 / squareness)
    zc = z0 + (z1 - z0) * 0.5
    rz = (z1 - z0) * 0.5
    z = np.where(s >= 0,
                 zc + rz * np.abs(s) ** dome,
                 zc - rz * np.abs(s) ** flat)
    return np.column_stack([np.full(n, x), cy + y, z])


def foot_body(scale_y=1.0):
    secs = [_section(x, cy * scale_y, w * scale_y, z0, z1)
            for x, cy, w, z0, z1 in STATIONS]
    m = loft(secs)
    m.process(validate=True)
    return m


def _toe_profile(f, radius, lift):
    """half-width, half-depth and centre height at fraction f along a toe.
    Shared with the nail beds, which have to sit ON this surface rather than
    near it - a nail modelled independently pokes out through the sides,
    which is what made the first ones look like rectangular plates."""
    t = radius * (1.0 - 0.14 * f)
    t *= np.sqrt(max(1e-4, 1.0 - f ** 4.2))                 # the tip
    t *= 1.0 + 0.075 * np.exp(-((f - 0.46) / 0.15) ** 2)    # the knuckle
    w = max(0.4, t * 1.08)
    h = max(0.34, t * 0.86)
    zc = 1.5 + radius * 0.90 + lift * f - 2.9 * f ** 2.2
    return w, h, zc


def toe(y, length, radius, splay, lift, x0=TOE_BREAK - 14.0):
    """
    One toe. Four things separate a toe from a sausage, and the first
    version had none of them.

      section   a toe is WIDER than it is deep, and flattened underneath
                where it stands on the ground - not a circle
      knuckle   a slight swelling at the interphalangeal joint, about
                45% along, which is the thing your eye actually reads
      curl      the tip drops. A toe that runs out straight and level
                looks like a peg
      tip       tapers over the last fifth rather than being a hemisphere
                stuck on the end of a cylinder
    """
    n = 36
    secs = []
    L = length + 15.0
    for k in range(n + 1):
        f = k / n
        w, h, zc = _toe_profile(f, radius, lift)
        secs.append(_section(x0 + L * f, 0.0, w, zc - h, zc + h,
                             n=44, squareness=2.4, flat=1.95, dome=0.90))
    m = loft(secs)
    m.apply_transform(rot(np.radians(splay), [0, 0, 1],
                          point=[x0, 0.0, 0.0]))
    m.apply_translation([0, y, 0])
    m.process(validate=True)
    return m


def malleolus(y, z, r, forward=0.0):
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = 1.25, 0.62, 1.0
    m.apply_transform(S)
    m.apply_translation([ANKLE_X + forward, y, z])
    return m


def ankle(z_top=96.0):
    """a short column above the talus, with the two ankle bones on it"""
    secs = []
    for k in range(24):
        f = k / 23
        z = 40.0 + (z_top - 40.0) * f
        # wide and low at the bottom so the union with the dorsum is a
        # blend rather than a collar dropped over it
        w = 31.5 - 8.5 * f ** 0.7
        d = 30.0 - 8.0 * f ** 0.7
        t = np.linspace(0, 2 * np.pi, 56, endpoint=False)
        c, s = np.cos(t), np.sin(t)
        secs.append(np.column_stack([
            ANKLE_X + d * np.sign(c) * np.abs(c) ** (2 / 2.6),
            -3.0 + w * np.sign(s) * np.abs(s) ** (2 / 2.6),
            np.full(56, z)]))
    col = loft(secs)
    col.process(validate=True)
    return col


def build_foot(with_ankle=True, with_toes=True):
    parts = [foot_body()]
    if with_toes:
        parts += [toe(*t[1:]) for t in TOES]
    if with_ankle:
        parts.append(ankle())
        parts.append(malleolus(21.0, 62.0, 9.0, forward=2.0))   # medial
        parts.append(malleolus(-27.0, 56.0, 8.0, forward=-2.0))  # lateral
    m = trimesh.boolean.union(parts, engine="manifold")
    m.process(validate=True)
    return m


if __name__ == "__main__":
    import time
    t0 = time.time()
    f = build_foot()
    print(f"built in {time.time() - t0:.0f}s")
    print(f"  tris {len(f.faces)}  watertight {f.is_watertight}")
    e = f.bounds[1] - f.bounds[0]
    print(f"  bbox {e[0]:.0f} x {e[1]:.0f} x {e[2]:.0f} mm"
          f"   vol {f.volume/1000:.0f} cm3")
    f.export("foot_anatomical.stl")


# --- the second pass of anatomy ---------------------------------------------
# The loft plus toes gets the SILHOUETTE right. What it does not have is any
# of the relief that tells you a foot is a bag of tendons and bones rather
# than a shoe last - and relief is the only thing you can still see once the
# whole object is one colour. So: the four extensor tendons standing off the
# dorsum, the knuckles they run to, the Achilles ridge down the back of the
# ankle, the nail beds, and a crease across the base of each toe.
#
# All of it is 1 to 3mm proud, which is deliberate. It stays under the 45
# degree rule everywhere, so none of it costs a millimetre of support, and
# it is far more than one layer, so all of it survives the slicer.

def _dorsum(x):
    xs = [s[0] for s in STATIONS]
    return float(np.interp(x, xs, [s[4] for s in STATIONS]))


def _rib(path, r0, r1, n=30, squash=0.70):
    """a rib swept along a path of (x, y, z) control points"""
    p = np.asarray(path, float)
    t = np.linspace(0, 1, n)
    u = np.linspace(0, 1, len(p))
    xs, ys, zs = (np.interp(t, u, p[:, i]) for i in range(3))
    a = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    secs = []
    for k in range(n):
        r = r0 + (r1 - r0) * t[k]
        # taper to nothing at both ends so the rib melts into the dorsum
        r *= np.sin(np.pi * np.clip(t[k], 0.02, 0.98)) ** 0.35
        secs.append(np.column_stack([np.full(30, xs[k]),
                                     ys[k] + r * np.cos(a),
                                     zs[k] + r * squash * np.sin(a)]))
    m = loft(secs)
    m.process(validate=True)
    return m


def tendons():
    """the four long extensors, fanning from the ankle to the toe bases"""
    out = []
    for name, y, L, r, splay, lift in TOES[:4]:
        x0, x1 = 74.0, TOE_BREAK - 12.0
        pts = []
        for f in np.linspace(0, 1, 7):
            x = x0 + (x1 - x0) * f
            # bunched on the centreline at the ankle, spread to its own toe
            yy = -2.5 + (y + 2.5) * f ** 1.5
            pts.append((x, yy, _dorsum(x) - 2.6))
        out.append(_rib(pts, 2.6, 3.4))
    return out


def knuckles():
    """
    The metatarsal heads. Long and very shallow, not round: they are the
    ENDS OF BONES running the length of the foot, and modelled as spheres
    they read as pebbles laid on the skin, which is exactly what the first
    version looked like.
    """
    out = []
    for name, y, L, r, splay, lift in TOES:
        x = TOE_BREAK - 20.0
        m = trimesh.creation.icosphere(subdivisions=3, radius=r * 0.86)
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = 2.30, 0.94, 0.30
        m.apply_transform(S)
        m.apply_translation([x, y * 0.94, _dorsum(x) - r * 0.62])
        m.apply_transform(rot(np.radians(splay * 0.6), [0, 0, 1],
                              point=[TOE_BREAK - 14.0, 0.0, 0.0]))
        out.append(m)
    return out


def achilles():
    """the tendon ridge down the back of the ankle, above the heel"""
    p = np.array([(19.0, 40.0), (17.0, 56.0), (15.5, 72.0), (15.0, 92.0)])
    t = np.linspace(0, 1, 26)
    u = np.linspace(0, 1, 4)
    xs, zs = np.interp(t, u, p[:, 0]), np.interp(t, u, p[:, 1])
    a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    secs = []
    for k in range(26):
        r = (7.2 - 2.6 * t[k]) * np.sin(np.pi * np.clip(t[k], .03, .99)) ** .3
        secs.append(np.column_stack([xs[k] + r * 0.62 * np.cos(a),
                                     -3.0 + r * np.sin(a),
                                     np.full(28, zs[k])]))
    m = loft(secs)
    m.process(validate=True)
    return m


def nail_beds():
    """
    Five nails: narrow almonds sunk into the toe so that only about eight
    tenths of a millimetre stands proud, and comfortably inside the toe's
    own width so no edge ever breaks out through the side.
    """
    out = []
    for name, y, L, r, splay, lift in TOES:
        x0 = TOE_BREAK - 14.0
        f = 0.76
        w, h, zc = _toe_profile(f, r, lift)
        n = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = r * 0.80, w * 0.56, 2.5
        n.apply_transform(S)
        n.apply_translation([x0 + (L + 15.0) * f, 0.0, zc + h - 1.7])
        n.apply_transform(rot(np.radians(splay), [0, 0, 1],
                              point=[x0, 0.0, 0.0]))
        n.apply_translation([0, y, 0])
        out.append(n)
    return out


def _slot(path_r, path_zf, x0, span, z_top=44.0, n=22, m=18):
    """
    A vertical slot with a rounded floor, swept along +x. Cutting a cleft
    with a BOX leaves square corners at the web, which is what made the
    first toes look machined; a rounded floor leaves the fillet a real
    cleft has.
    """
    secs = []
    for k in range(n + 1):
        f = k / n
        r = path_r(f)
        zf = path_zf(f)
        t = np.linspace(np.pi, 2 * np.pi, m)
        ring = [(r * np.cos(a), zf + r * np.sin(a)) for a in t]
        ring += [(r, z_top), (-r, z_top)]
        secs.append(np.column_stack([np.full(len(ring), x0 + span * f),
                                     [p[0] for p in ring],
                                     [p[1] for p in ring]]))
    g = loft(secs)
    g.process(validate=True)
    return g


def toe_clefts():
    """
    The cleft between each pair of toes: shallow at the web, running down
    almost to the sole at the front, and widening as it goes - which is the
    shape of the gap between two toes that touch at the base and separate
    toward the tips.
    """
    cuts = []
    for a, b in zip(TOES[:-1], TOES[1:]):
        ym = (a[1] + b[1]) / 2
        sm = (a[4] + b[4]) / 2
        rr = min(a[3], b[3])
        top = 1.5 + 1.76 * rr
        # the floor starts ABOVE the toe, so the cleft fades in at the web
        # instead of beginning with a square wall - which is what left a
        # notch across the root of every toe in the first version
        cuts.append(_slot(lambda f, rr=rr: 0.12 * rr + 0.14 * rr * f,
                          lambda f, top=top: top * (1.02 - 0.86 * f ** 1.15),
                          TOE_BREAK - 10.0, 58.0))
        cuts[-1].apply_transform(rot(np.radians(sm), [0, 0, 1],
                                     point=[TOE_BREAK - 14.0, 0.0, 0.0]))
        cuts[-1].apply_translation([0, ym, 0])
    return cuts


def toe_creases():
    """a soft crease across the top of each toe at its root - a rolled
    cylinder taking a millimetre off, not a box taking a slice out"""
    out = []
    for name, y, L, r, splay, lift in TOES:
        c = trimesh.creation.cylinder(radius=r * 0.46, height=r * 2.1,
                                      sections=28)
        c.apply_transform(rot(np.radians(90), [1, 0, 0]))
        c.apply_translation([TOE_BREAK + 1.0, 0.0,
                             1.5 + r * 0.90 + r * 0.86 + r * 0.47])
        c.apply_transform(rot(np.radians(splay), [0, 0, 1],
                              point=[TOE_BREAK - 14.0, 0.0, 0.0]))
        c.apply_translation([0, y, 0])
        out.append(c)
    return out


def detailed_foot():
    """the foot with its second pass of anatomy on it"""
    m = trimesh.boolean.union(
        [build_foot()] + tendons() + knuckles() + [achilles()] + nail_beds(),
        engine="manifold")
    # no creases across the toes any more. A cylinder rolled across a toe
    # leaves a hard step, and at this scale that reads as a machining mark
    # rather than as a knuckle - the swelling in _toe_profile does the job
    for c in toe_clefts():
        m = trimesh.boolean.difference([m, c], engine="manifold")
    return m
