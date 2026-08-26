"""
Paduka - the web visualiser.

A page you can turn the object around in and pull apart, built to be one
self-contained file. Three decisions make that possible.

**The pattern is a texture, not geometry.** On the print files each dye is a
0.7mm slab, which costs a million triangles. A viewer does not need that: the
drawing already lives in an unrolled (u, v) space, so it bakes straight to an
image and the low-poly modules carry uv coordinates into it. 200k triangles
instead of 3.5 million.

**The texture is an INDEX map, not a colour map.** Each texel holds 0-3:
ground, outline, fill, detail. The shader looks the number up in a four-entry
palette passed as uniforms, so switching colourway is four numbers rather
than a second megabyte-and-a-half image, and it happens in one frame.

**The seam is unwrapped per triangle.** A cylindrical unrolling puts a branch
cut down the back of the leg, and any triangle straddling it would smear the
whole repeat across itself. Because the geometry is sent unindexed, each
triangle's three angles can be taken on the branch nearest its own centre,
which closes the seam without duplicating a single vertex by hand.
"""

import base64, io, json, zlib
import numpy as np
import trimesh

import paduka_stack as S
import paduka_sandal as SA
import paduka_modular as MOD
import paduka_pattern as PP

TEX_W, TEX_H = 1200, 3456
BARE_UV = (163.0, 477.0)      # a corner of the chart kept permanently blank
LEVELS = {"ground": 0, "outline": 1, "fill": 2, "detail": 3}


# ------------------------------------------------------------------ texture ---

def bake_texture(way="madder"):
    """
    The repeat, rendered once into an index image. Antialiasing is off and
    the result is snapped back to four levels: a blended edge between
    'outline' and 'fill' would decode as 'nothing', which is exactly the
    kind of bug that shows up as sparkle all over a pattern.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from preview2d import draw

    pat = S.drawing(way)
    dpi = 100.0
    fig = plt.figure(figsize=(TEX_W / dpi, TEX_H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0, 0, 0))
    for name in ("outline", "fill", "detail"):
        g = pat.get(name)
        if g is not None and not g.is_empty:
            v = LEVELS[name] / 3.0
            draw(ax, g, fc=(v, v, v), ec="none", antialiased=False)
    ax.set_xlim(*S.U_SPAN); ax.set_ylim(*S.V_SPAN)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, facecolor=(0, 0, 0))
    plt.close(fig)

    from PIL import Image
    im = Image.open(buf).convert("L").resize((TEX_W, TEX_H), Image.NEAREST)
    a = np.asarray(im).astype(np.int16)
    idx = np.clip(np.rint(a / 85.0), 0, 3).astype(np.uint8)
    # the left-hand column is outside the used u range on every chart, so it
    # is free real estate: paint it solid 'outline' and point the welt at it
    idx[:, :6] = LEVELS["outline"]
    idx[:14, -14:] = LEVELS["ground"]     # the blank corner BARE_UV points at
    out = io.BytesIO()
    Image.fromarray((idx * 85).astype(np.uint8)).save(out, format="PNG",
                                                      optimize=True)
    return out.getvalue()


# ----------------------------------------------------------------- geometry ---

# Where the chart changes hands. The object is two tubes lying along two
# different axes, and the cut between them has to fall where the geometry
# actually changes direction - the ankle parting plane at z=66, not the
# malleoli at 86. Between those two heights the ankle is already a vertical
# column, and unrolling a vertical column about a horizontal axis stretches
# every motif on it into a smear. That is the whole of the warping that used
# to show on the ankle.
UV_SPLIT = 66.0
FLAT = 0.80        # cos of the angle at which a face stops being tube wall


def face_uv_unwrapped(mesh):
    """per-triangle-vertex uv, with the cylindrical branch cut closed"""
    tri = np.asarray(mesh.triangles, float)         # (F, 3, 3)
    c = tri.mean(axis=1)
    n = np.asarray(mesh.face_normals, float)
    F = len(tri)
    u = np.empty((F, 3)); v = np.empty((F, 3))

    def unwrap(a, ac):
        return ac[:, None] + (a - ac[:, None] + np.pi) % (2 * np.pi) - np.pi

    hi = c[:, 2] >= UV_SPLIT
    if hi.any():
        ac = np.arctan2(c[hi, 1] - S.AXIS_Y, c[hi, 0] - S.AXIS_X)
        a = np.arctan2(tri[hi][:, :, 1] - S.AXIS_Y, tri[hi][:, :, 0] - S.AXIS_X)
        u[hi] = unwrap(a, ac) * S.R_REF
        v[hi] = tri[hi][:, :, 2] - S.LIMB_DZ
    lo = ~hi
    if lo.any():
        ac = np.arctan2(c[lo, 2] - S.FOOT_ZC, c[lo, 1] - S.AXIS_Y)
        a = np.arctan2(tri[lo][:, :, 2] - S.FOOT_ZC, tri[lo][:, :, 1] - S.AXIS_Y)
        u[lo] = unwrap(a, ac) * S.FOOT_R
        v[lo] = S.FOOT_V0 - (tri[lo][:, :, 0] - S.AXIS_X)

    # --- everything that is not tube wall goes bare -----------------------
    # A cylindrical unwrap is only honest on the wall of its own cylinder.
    # Project a face that faces along the axis instead of across it and its
    # whole area lands on a line of the chart, so whatever was drawn there
    # arrives as a radial smear. Those faces are the seam rings, the collar
    # shoulders, the recess floor, the cut ends: machined flats, every one
    # of them, and on the printed object every one is held at the ground
    # filament anyway because a mating face must be a single material. So
    # the render agrees with the print by sending them to a corner of the
    # texture that is guaranteed to be blank.
    # The test is against each chart's OWN axis, which is the part I got
    # wrong first time: the limb's tube runs up Z, so its flats face along
    # Z; the foot's tube runs fore-and-aft along X, so the top of the foot
    # is not a flat at all - it is the wall of its own cylinder, and it
    # should be printed on. Only the heel and the cut ends face along X.
    bare = c[:, 0] > S.AXIS_X + 156.0            # the toes, off the end
    bare |= hi & (np.abs(n[:, 2]) > FLAT)        # limb: seams and collars
    bare |= (~hi) & (np.abs(n[:, 0]) > FLAT)     # foot: the heel and cuts
    u[bare] = BARE_UV[0]
    v[bare] = BARE_UV[1]
    return np.stack([u, v], axis=-1)


LO = np.array([-10.0, -60.0, -5.0])
HI = np.array([270.0, 55.0, 430.0])


def pack(mesh, uv=None):
    tri = np.asarray(mesh.triangles, float).reshape(-1, 3)
    p = np.clip((tri - LO) / (HI - LO), 0, 1)
    pos = (p * 65535).astype("<u2")
    if uv is None:
        return pos.tobytes(), b""
    q = uv.reshape(-1, 2).astype(float)
    q[:, 0] = (q[:, 0] - S.U_SPAN[0]) / (S.U_SPAN[1] - S.U_SPAN[0])
    q[:, 1] = (q[:, 1] - S.V_SPAN[0]) / (S.V_SPAN[1] - S.V_SPAN[0])
    return pos.tobytes(), (np.clip(q, 0, 1) * 65535).astype("<u2").tobytes()


STATS = {}


def build_parts():
    """every drawn piece, with its module, its shading mode and its buffers"""
    parts, blobs = [], []

    def add(name, module, mesh, mode, uv=None):
        pb, ub = pack(mesh, uv)
        parts.append(dict(name=name, module=module, mode=mode,
                          n=len(mesh.faces) * 3,
                          pos=len(blobs), uv=(len(blobs) + 1) if ub else -1))
        blobs.append(pb)
        if ub:
            blobs.append(ub)

    T = SA.build_tree()
    sole = T["sole"]
    plate = trimesh.util.concatenate(MOD._prism(sole, 0.0, MOD.PLATE_T))
    rail = MOD.dovetail(MOD.RAIL_W, MOD.RAIL_H, MOD.RAIL_X1 - MOD.RAIL_X0,
                        MOD.RAIL_X0, PP.centre_at(sole, 124.0))
    rail.apply_translation([0, 0, MOD.PLATE_T])
    base = trimesh.boolean.union([plate, rail, MOD.toe_post()],
                                 engine="manifold")
    add("sole", "paduka", base, 5)
    e = base.bounds[1] - base.bounds[0]
    STATS["paduka"] = dict(mm=[round(float(x)) for x in e],
                           cm3=round(base.volume / 1000, 1))
    for k in ("outline", "fill", "detail"):
        ps = []
        for h, g in T["bodies"][k].items():
            if not g.is_empty:
                ps += MOD._prism(g, MOD.PLATE_T - 0.4, MOD.PLATE_T + h)
        if ps:
            add(k, "paduka", trimesh.util.concatenate(ps), LEVELS[k] + 1)

    for nm, m in S.modules().items():
        e = m.bounds[1] - m.bounds[0]
        STATS[nm] = dict(mm=[round(float(x)) for x in e],
                         cm3=round(m.volume / 1000, 1))
        # the foot gets a finer mesh than the limb. Its welt is a height
        # threshold rather than a line in the drawing, so the edge is only
        # ever as smooth as the triangles that carry it
        d = m.subdivide_to_size(3.2 if nm in ("foot", "ankle") else 6.0)
        add(nm, nm, d, 0, face_uv_unwrapped(d))
    return parts, blobs


# -------------------------------------------------------------------- page ---

def build(way_list=("madder", "indigo"), out="paduka_visualiser.html"):
    parts, blobs = build_parts()
    off, cur = [], 0
    for b in blobs:
        off.append(cur); cur += len(b)
    raw = b"".join(blobs)
    packed = base64.b64encode(zlib.compress(raw, 9)).decode()
    tex = base64.b64encode(bake_texture(way_list[0])).decode()

    meta = dict(
        parts=parts, offsets=off, lo=LO.tolist(), hi=HI.tolist(),
        modules=["paduka"] + list(S.NAMES),
        joints=[S.PLATE_T] + list(S.JOINT), top=S.TOP_Z,
        ways={k: {n: [round(float(x), 4) for x in c]
                  for n, c in v.items()} for k, v in S.WAYS.items()},
        wood=[0.560, 0.380, 0.215],
        stats=STATS,
        tri=sum(p["n"] for p in parts) // 3,
    )
    html = PAGE.replace("__META__", json.dumps(meta)) \
               .replace("__GEO__", packed).replace("__TEX__", tex)
    with open(out, "w") as f:
        f.write(html)
    print(f"{out}  {len(html)/1e6:.2f} MB   "
          f"{sum(p['n'] for p in parts)//3} triangles")
    return out


PAGE = ""   # filled in by paduka_web_page.py

if __name__ == "__main__":
    from paduka_web_page import PAGE as P
    PAGE = P
    build()
