"""
Paduka - the modular pair: an ornamented sandal, and a foot that clips into it.

The architecture the project arrived at, and the one worth submitting.

The earlier versions put the ornament ON the prosthetic foot, which meant
every new pattern was a new foot: a structural component reprinted for a
cosmetic reason. This inverts it. The **paduka is the shoe** - a thick
ornamented sole, the centrepiece, printed in four filaments and made in as
many colourways as you like. The **foot** is one component, printed once,
that clips into any of them.

You own one foot and several paduka. Which is how footwear has always
worked, and is the first version of this idea that a person with one leg
would actually recognise as a wardrobe rather than as a medical fitting.

The interface is the part worth looking at closely. A paduka is held on
the foot by a post gripped between the great toe and the second - the
khadau knob, the oldest detail in Indian footwear. Here **that post is the
locating pin**: it passes up through a cleft in the foot and registers the
two parts against each other. A dovetail rail under the foot takes the
fore-aft load, and a single thumbscrew at the heel stops it sliding back
off. No tools, one hand, and the heritage detail is doing the engineering
rather than being quoted at you.

  paduka   wood ground + white / blue / black block-print paisley
  foot     one colour, structural, standard 30mm pyramid on top
"""

import numpy as np
import trimesh
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely import affinity

import paduka_pattern as PP
from palampore import clean, largest
import paduka_block as B

PLATE_T   = 14.0      # the sandal sole
RELIEF_LO = 1.15
RELIEF_HI = 1.75

FOOT_X = 7.0          # the foot is 245 long on a 260 sandal: centre it
POST_X, POST_Y = 227.0, 13.5
POST_D, POST_H = 13.0, 21.0
RAIL_X0, RAIL_X1 = 60.0, 190.0
RAIL_W, RAIL_H = 21.0, 5.0   # narrow, and it lands in the arch - which is
                             # exactly where a shoe puts its shank
CLEAR = 0.35          # print clearance on every mating face


def _sole():
    return PP.sole_outline()


def _prism(poly, z0, z1):
    out = []
    ps = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for p in ps:
        if p.area < 1.0:
            continue
        try:
            m = trimesh.creation.extrude_polygon(p.simplify(0.05), abs(z1 - z0))
        except Exception:
            continue
        m.apply_translation([0, 0, min(z0, z1)])
        out.append(m)
    return out


def dovetail(w, h, length, x0, cy, clearance=0.0):
    """
    A dovetail rail lying along +x. Undercut 15 degrees a side, which is
    self-supporting on a printer and will not lift out under load - the
    foot can only leave the sandal by sliding backwards off the heel.
    """
    from shapely.geometry import Polygon as _P
    t = h * np.tan(np.radians(15.0))
    lo = w / 2 - t + clearance
    hi = w / 2 + clearance
    # drawn in (y, z), then the whole prism is cycled so its extrusion
    # axis becomes +x: polygon-x -> world-y, polygon-y -> world-z
    sec = _P([(-lo, 0), (lo, 0), (hi, h), (-hi, h)])
    m = trimesh.creation.extrude_polygon(sec, length)
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=float)
    m.apply_transform(T)
    m.apply_translation([x0, cy, 0])
    return m


def toe_post(clearance=0.0):
    """the khadau knob: a stem with a mushroom head, and the locating pin"""
    r = POST_D / 2 + clearance
    stem = trimesh.creation.cylinder(radius=r, height=POST_H, sections=48)
    stem.apply_translation([POST_X, POST_Y, PLATE_T + POST_H / 2])
    head = trimesh.creation.icosphere(subdivisions=2, radius=r * 1.32)
    head.apply_translation([POST_X, POST_Y, PLATE_T + POST_H])
    return trimesh.util.concatenate([stem, head])


def build_paduka():
    """the sandal: plate, ornament in three dye bodies, post and rail"""
    sole = _sole()
    lay = B.build_tree(shell=True)

    plate = trimesh.util.concatenate(_prism(sole, 0.0, PLATE_T))
    rail = dovetail(RAIL_W, RAIL_H, RAIL_X1 - RAIL_X0, RAIL_X0,
                    PP.centre_at(sole, 124.0))
    rail.apply_translation([0, 0, PLATE_T])
    base = trimesh.boolean.union([plate, rail, toe_post()], engine="manifold")

    bodies = {}
    for name in (B.WHITE, B.BLUE, B.BLACK):
        parts = []
        for h, g in lay["bodies"][name].items():
            if g.is_empty:
                continue
            parts += _prism(g, PLATE_T - 0.5, PLATE_T + h)
        if parts:
            bodies[name] = trimesh.util.concatenate(parts)

    keep = [m for m in bodies.values()]
    if keep:
        base = trimesh.boolean.difference([base] + keep, engine="manifold")
    return base, bodies


def build_foot():
    """
    The component you only buy once. The anatomical foot from paduka_foot,
    seated on the sandal, with the dovetail channel and the post cleft cut
    out of its underside and a standard pyramid on the ankle.
    """
    import paduka_foot as PF
    sole = _sole()
    foot = PF.build_foot()
    foot.apply_translation([FOOT_X, 0.0, PLATE_T])

    cuts = [dovetail(RAIL_W, RAIL_H, RAIL_X1 - RAIL_X0 + 60, RAIL_X0 - 30,
                     PP.centre_at(sole, 124.0), clearance=CLEAR)]
    cuts[0].apply_translation([0, 0, PLATE_T - 0.01])
    cuts.append(toe_post(clearance=CLEAR))
    # the post rises in the cleft between the great toe and the second, and
    # the slot is open forwards so the foot slides on rather than dropping in
    slot = trimesh.creation.box(extents=[80, POST_D + 2 * CLEAR, 70])
    slot.apply_translation([POST_X + 32, POST_Y, PLATE_T + 26])
    cuts.append(slot)

    foot = trimesh.boolean.difference([foot] + cuts, engine="manifold")

    zt = foot.bounds[1][2]
    land = trimesh.creation.box(extents=[44, 44, 6])
    land.apply_translation([PF.ANKLE_X + FOOT_X, -3, zt - 2])
    pyr = trimesh.creation.cone(radius=17, height=22, sections=4)
    pyr.apply_translation([PF.ANKLE_X + FOOT_X, -3, zt + 1])
    return trimesh.boolean.union([foot, land, pyr], engine="manifold")


if __name__ == "__main__":
    import time
    t0 = time.time()
    pad, bodies = build_paduka()
    foot = build_foot()
    print(f"built in {time.time() - t0:.0f}s")
    print(f"  paduka base  tris {len(pad.faces):6d}  vol {pad.volume/1000:6.1f} cm3")
    for k, v in bodies.items():
        print(f"  {k:11} tris {len(v.faces):6d}  vol {v.volume/1000:6.1f} cm3")
    print(f"  foot         tris {len(foot.faces):6d}  vol {foot.volume/1000:6.1f} cm3")
    pad.export("mod_paduka_base.stl")
    for k, v in bodies.items():
        v.export(f"mod_paduka_{k}.stl")
    foot.export("mod_foot.stl")
