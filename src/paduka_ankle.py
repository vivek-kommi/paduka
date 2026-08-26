"""
Paduka - the ankle.

The stack had a rigid ankle: the foot's collar and the shank's socket, pinned.
That is fine for a display piece and wrong for a foot. A foot that cannot
move at the ankle has to be walked on by rolling over its toe, which is what
makes a cheap prosthetic foot read as a peg.

So the foot gets the joint a real single-axis foot has, and gets it in the
same place a real one does: **a transverse pin through the malleoli, with a
bumper in front of it and a bumper behind it.**

  plantarflexion   the REAR bumper compresses as the heel lands, so the
                   foot comes flat instead of slapping
  dorsiflexion     the FRONT bumper compresses as you roll forward over
                   it, and pushes back as you leave - which is where the
                   energy return in a cheap foot comes from
  the pin          takes the load; the bumpers set the travel and the feel

Four printed pieces where there was one: foot, ankle block, and two bumpers.
The bumpers are the tuning - print them soft for a slow, forgiving foot and
firm for a brisk one, and they are 4cc each, so a set is minutes.

The axis is at z = 74, which is the height of the malleoli on this foot,
because the malleoli ARE the axis - they are the ends of the bones the pin
replaces. Putting the joint anywhere else makes the ankle look like a
mechanism bolted to an ankle rather than the ankle itself.
"""

import numpy as np
import trimesh
from shapely.geometry import Polygon

AX_Z = 74.0          # the flex axis: malleolus height
ANK_CUT = 66.0       # the parting plane - the ankle block's underside
ANK_GAP = 6.0        # working clearance between the two, at neutral
R_BOSS = 16.0        # the boss's rounded head, concentric with the pin
BOSS_T = 22.0        # how thick the boss is across
ANK_CLEAR = 0.35
PIN2_D = 6.3         # for a 6mm pin
BUMP_X = 25.0        # how far fore and aft of the axis the bumpers sit
BUMP_W, BUMP_D = 17.0, 21.0
POCKET = 1.6         # how far a bumper is captured at each end
COL_RX, COL_RY = 30.0, 31.0   # the ankle column's plan half-extents
SWING = 8.5         # how far the column sweeps at its top over full travel,
                     # and therefore how far the foot is cut back around it


def _sweep(poly, thick, cy):
    """
    Extrude a profile drawn in (x, z) along the y axis - the pin's axis.
    trimesh extrudes a plan into +z, so the result is turned a quarter turn
    about x: the profile's second coordinate becomes height, and the
    extrusion becomes width across the foot.
    """
    m = trimesh.creation.extrude_polygon(poly, thick)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0]))
    m.apply_translation([0, cy + thick / 2, 0])
    return m


def boss(ax_x, cy, grow=0.0, z_bot=50.0, n=64):
    """
    The tongue that rises out of the foot and carries the pin. Its head is a
    half-disc CONCENTRIC with the pin, so the fork closes round it at a
    constant gap and the joint has the same clearance at every angle - which
    a square-topped boss does not.
    """
    r = R_BOSS + grow
    a = np.linspace(0, np.pi, n)
    ring = [(ax_x + r * np.cos(t), AX_Z + r * np.sin(t)) for t in a]
    ring += [(ax_x - r, z_bot), (ax_x + r, z_bot)]
    return _sweep(Polygon(ring), BOSS_T + 2 * grow, cy - grow)


def fork_slot(ax_x, cy):
    """the negative of the boss, open downward, cut out of the ankle block"""
    return boss(ax_x, cy, grow=ANK_CLEAR, z_bot=20.0)


def pin_bore(ax_x, cy, d=PIN2_D):
    m = trimesh.creation.cylinder(radius=d / 2, height=220, sections=48)
    m.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(90), [1, 0, 0]))
    m.apply_translation([ax_x, cy, AX_Z])
    return m


def _pocket(ax_x, cy, sign, z0, z1, grow=0.0):
    b = trimesh.creation.box(extents=[BUMP_W + 2 * grow,
                                      BUMP_D + 2 * grow, z1 - z0])
    b.apply_translation([ax_x + sign * BUMP_X, cy, (z0 + z1) / 2])
    return b


def column(ax_x, cy, grow=0.0, z0=None, n=72):
    """
    The plan of the ankle column: an oval prism about the pin axis.

    The parting plane cannot simply be a plane. The foot's instep is still
    80mm long at this height, so a flat cut hands the ankle block a long
    thin flange of dorsum that is fragile, ugly, and nothing like an ankle.
    Instead the ankle block is the COLUMN, and the instep stays with the
    foot - with a recess cut round the column for it to swing in. Which is
    what a real ankle is: a column standing in a socket in the top of the
    foot.
    """
    z0 = ANK_CUT - ANK_GAP - 12.0 if z0 is None else z0
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rx, ry = COL_RX + grow, COL_RY + grow
    ring = [(ax_x + rx * np.cos(a), cy + ry * np.sin(a)) for a in t]
    m = trimesh.creation.extrude_polygon(Polygon(ring), 400.0)
    m.apply_translation([0, 0, z0])
    return m


def shave(ax_x, cy):
    """
    The working gap: a slab taken out of the top of the foot inside the
    column's recess, so the joint has room to move. Outside the recess the
    instep is untouched.
    """
    b = trimesh.creation.box(extents=[400, 400, ANK_GAP])
    b.apply_translation([ax_x, cy, ANK_CUT - ANK_GAP / 2])
    b = trimesh.boolean.intersection(
        [b, column(ax_x, cy, grow=SWING)], engine="manifold")
    return trimesh.boolean.difference(
        [b, boss(ax_x, cy, grow=ANK_CLEAR)], engine="manifold")


def bumper():
    """
    One bumper. A block chamfered at both ends so it seats into its pockets
    without a lip to catch on, 9.2mm tall for a 6mm working gap with 1.6mm
    captured at each end.

    Print these in TPU if you have it. In PLA, two walls and 12% gyroid
    gives a surprisingly good spring - and a bumper is 3cc, so print three
    stiffnesses and walk on the one you like. They are the tuning.
    """
    from paduka_leg import loft
    h = ANK_GAP + 2 * POCKET
    w, d = (BUMP_W - 0.6) / 2, (BUMP_D - 0.6) / 2
    secs = []
    for z, k in ((0.0, 0.86), (1.3, 1.0), (h - 1.3, 1.0), (h, 0.86)):
        secs.append(np.array([[-w * k, -d * k, z], [w * k, -d * k, z],
                              [w * k, d * k, z], [-w * k, d * k, z]]))
    m = loft(secs)
    m.process(validate=True)
    return m


def split(foot, ax_x, cy):
    """
    Turn one foot module into a working ankle.

      foot    everything below the parting plane, plus the instep that
              stands above it outside the column's recess, plus the boss
      ankle   the column above the parting plane, with the fork cut into it

    Both get the pin bore and a pocket for each bumper.
    """
    from trimesh.intersections import slice_mesh_plane

    lo = slice_mesh_plane(foot, [0, 0, -1], [0, 0, ANK_CUT], cap=True)
    hi = slice_mesh_plane(foot, [0, 0, 1], [0, 0, ANK_CUT], cap=True)

    col = column(ax_x, cy)
    col_c = column(ax_x, cy, grow=SWING)
    ankle = trimesh.boolean.intersection([hi, col], engine="manifold")
    rim = trimesh.boolean.difference([hi, col_c], engine="manifold")

    zf = ANK_CUT - ANK_GAP
    lo = trimesh.boolean.difference([lo, shave(ax_x, cy)], engine="manifold")
    lo = trimesh.boolean.union([lo, rim, boss(ax_x, cy)], engine="manifold")
    for s_ in (+1, -1):
        lo = trimesh.boolean.difference(
            [lo, _pocket(ax_x, cy, s_, zf - POCKET, zf, grow=ANK_CLEAR)],
            engine="manifold")
    lo = trimesh.boolean.difference([lo, pin_bore(ax_x, cy)],
                                    engine="manifold")

    ankle = trimesh.boolean.difference([ankle, fork_slot(ax_x, cy)],
                                       engine="manifold")
    for s_ in (+1, -1):
        ankle = trimesh.boolean.difference(
            [ankle, _pocket(ax_x, cy, s_, ANK_CUT, ANK_CUT + POCKET,
                            grow=ANK_CLEAR)], engine="manifold")
    ankle = trimesh.boolean.difference([ankle, pin_bore(ax_x, cy)],
                                       engine="manifold")
    return lo, ankle


def hidden(c, ax_x, cy):
    """
    Every face this joint creates is buried when it is together: the parting
    faces, the working gap, the boss, the fork bore and both pockets. They
    are held at the ground filament like every other mating face.
    """
    z, x, y = c[:, 2], c[:, 0], c[:, 1]
    zf = ANK_CUT - ANK_GAP
    r = np.hypot((x - ax_x) / (COL_RX + SWING + 1.0),
                 (y - cy) / (COL_RY + SWING + 1.0))
    sel = (z >= zf - POCKET - 0.4) & (z <= ANK_CUT + POCKET + 0.4) & (r < 1.25)
    sel |= ((z <= AX_Z + R_BOSS + 1.5) & (z >= zf - 1.0)
            & (np.abs(y - cy) <= BOSS_T / 2 + 2.0)
            & (np.abs(x - ax_x) <= R_BOSS + 2.0))
    return sel
