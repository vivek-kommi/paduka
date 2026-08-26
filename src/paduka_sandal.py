"""
Paduka - the sandal, re-cut with the paisley blocks.

Same three blocks as the limb, so a paduka and a limb printed together read
as one cloth. The differences are the ones a border cloth actually has:

  the field   laid out in the sole's own plane rather than unrolled, because
              the sole IS flat - and set at a scale that puts about five
              butas across the waist
  the hashiya a run of small butas standing INWARD off the edge, following
              the sole outline rather than a straight line. A border of
              paisleys lying on their side reads as a row of commas.
  relief      the outline and fill blocks sit 1.15mm proud, the detail block
              1.75mm. Two heights, not three - every extra height is another
              twelve tool changes at the prime tower.
"""

import numpy as np
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from shapely import affinity
import shapely

import paduka_pattern as PP
from paduka_pattern import sole_outline
from palampore import clean, largest, place
import paduka_paisley as PZ
from paduka_tree import PLATE_T, pedestal_footprints

FIELD_IN = 9.5
BORDER_IN = 3.2
H_UNDER = 1.15
H_OVER = 1.75

OUTLINE, FILL, DETAIL = "outline", "fill", "detail"
PITCH = (44.0, 46.0)
W_BIG = 26.0


def _lg(g):
    return largest(g) if not g.is_empty else g


def sandal_field(sole, field):
    """
    The field is laid ALONG the sole, not across a rectangle and trimmed.

    A sandal is 95mm at the ball and 55mm at the waist, so a repeat on a
    fixed grid either shatters in the waist or wastes the ball. A block
    printer working a shaped ground does the obvious thing instead: walks
    the centre line and stamps the biggest block that fits where he is. So
    the butas swell over the ball and the heel and run small through the
    arch, which is the shape of the shoe showing through its ornament.
    """
    O, F, D = [], [], []
    stamped = []
    rng = np.random.default_rng(4)
    x = 14.0
    k = 0
    while x < 252.0:
        cy = PP.centre_at(sole, x)
        avail = PP.width_at(sole, x) - 2 * FIELD_IN
        # the buta stands UPRIGHT, across the sandal, the way a border buta
        # stands on a cloth. Its height is about 1.72 times its width, so it
        # is the sole's WIDTH that sets the size, not its length.
        w = float(np.clip(round(avail / 1.72 * 0.86 / 2) * 2, 14.0, 26.0))
        blocks = PZ.pool(w, n=3)[k % 3]
        gs = [affinity.rotate(
                  affinity.scale(g, -1, 1, origin=(0, 0)) if k % 2 else g,
                  float(rng.uniform(-3, 3)), origin=(0, 0))
              for g in blocks]
        raw = unary_union([g for g in gs if not g.is_empty])
        # centre the stamp on the station by its own bounding box - the
        # buta's origin is its base, not its middle
        bb = raw.bounds
        dx = x - (bb[0] + bb[2]) / 2
        dy = cy - (bb[1] + bb[3]) / 2
        gs = [affinity.translate(g, dx, dy) for g in gs]
        whole = affinity.translate(raw, dx, dy)
        if field.contains(whole):
            for g, dst in zip(gs, (O, F, D)):
                if not g.is_empty:
                    dst.append(g)
            stamped.append(whole)
            x += w * 1.08
            k += 1
        else:
            x += 4.0
    guard = clean(unary_union(stamped)) if stamped else Polygon()

    # rosettes wherever the row leaves room
    gx, gy = np.meshgrid(np.arange(6, 256, 4.0), np.arange(-52, 48, 4.0))
    pts = shapely.points(np.column_stack([gx.ravel(), gy.ravel()]))
    inside = shapely.contains(field.buffer(-1.0), pts)
    pts = pts[inside]
    xy = shapely.get_coordinates(pts)
    room = np.minimum(shapely.distance(pts, guard),
                      shapely.distance(pts, field.boundary))
    ph = [PZ.phool(d, seed=i) for i, d in enumerate((6.5, 8.0, 9.5, 11.0))]
    covered = guard
    for _ in range(400):
        i = int(np.argmax(room))
        r = float(room[i]) - PZ.REG
        if r < 6.0:
            break
        kk = int(np.clip((2 * r * 0.88 - 6.5) / 1.5, 0, 3))
        a = affinity.translate(ph[kk][0], *xy[i])
        c2 = affinity.translate(ph[kk][1], *xy[i])
        O.append(a); D.append(c2)
        covered = clean(unary_union([covered, a, c2]))
        room = np.minimum(room, shapely.distance(pts, a.union(c2)))

    outline = clean(unary_union([g for g in O if not g.is_empty]))
    fill = clean(unary_union([g for g in F if not g.is_empty]))
    detail = clean(unary_union([g for g in D if not g.is_empty]))
    ink = clean(unary_union([outline, fill, detail]))
    b = sole.bounds
    dots = PZ.chownk(b[0], b[2], b[1], b[3], ink, keep=field)
    detail = clean(unary_union([detail, dots]))
    return outline, fill, detail


def build_tree(shell=True):
    sole = sole_outline()
    field = _lg(clean(sole.buffer(-FIELD_IN)))
    O, F, D = sandal_field(sole, field)

    # --- the two rules that frame the field ------------------------------
    rule = LineString(_lg(sole.buffer(-BORDER_IN)).exterior).buffer(
        1.5, cap_style=1)
    bO = [clean(rule.intersection(sole.buffer(-0.6)))]
    rule2 = LineString(_lg(sole.buffer(-(BORDER_IN + 5.2))).exterior).buffer(
        0.95, cap_style=1)
    bD = [clean(rule2.intersection(sole))]
    bF = []

    out = {}
    for name, fg, bg in ((OUTLINE, O, bO), (FILL, F, bF), (DETAIL, D, bD)):
        g = fg
        g = clean(unary_union([g] + [x for x in bg if not x.is_empty]))
        # no final clip to the sole. Everything here was placed by a
        # containment test, so a clip can only ever cut something that was
        # already whole - which is exactly the fault being fixed.
        out[name] = PZ._ok(g, area=2.0)

    # two relief heights only: the detail block stands proudest, as the last
    # block stamped does on cloth
    bodies = {OUTLINE: {H_UNDER: out[OUTLINE]},
              FILL: {H_UNDER: out[FILL]},
              DETAIL: {H_OVER: out[DETAIL]}}
    ped_h, ped_f = pedestal_footprints(sole)
    return dict(sole=sole, field=field, bodies=bodies, holes=Polygon(),
                ped_heel=ped_h, ped_fore=ped_f,
                bolt_cy=PP.centre_at(sole, 124.0), n_hole=0)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from preview2d import draw
    T = build_tree()
    PAL = {OUTLINE: "#131A2A", FILL: "#A82B22", DETAIL: "#C9A24A"}
    fig, ax = plt.subplots(figsize=(18, 7.4))
    draw(ax, T["sole"], fc="#B08148")
    for k in (OUTLINE, FILL, DETAIL):
        for g in T["bodies"][k].values():
            draw(ax, g, fc=PAL[k])
    ax.set_xlim(-6, 266); ax.set_ylim(-56, 52); ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig("_sandal.png", dpi=150, bbox_inches="tight", facecolor="white")
    for k in (OUTLINE, FILL, DETAIL):
        print(k, round(sum(g.area for g in T["bodies"][k].values())), "mm2")
