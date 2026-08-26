"""
Paduka - the print files.

Turning the design into six printable pieces. Two problems to solve.

**The ink needs thickness.** On screen the pattern is carried per face -
ink has no thickness, which is true of cloth and useless to a nozzle. So
each dye's faces are lifted off the surface into a 0.7mm SLAB: the shell of
selected triangles, a copy of it pushed out along the vertex normals, and a
wall closing the boundary between them. Because the slab is built from the
faces that were already selected, it follows the surface exactly - no
projection, so nothing streaks down a steep face, which is the failure that
wrecked two earlier attempts at putting pattern on a curved foot. 0.7mm is
also, not coincidentally, what a wooden block leaves on cotton.

**Each piece has to fit a 256mm bed and print without a forest.** The two
limb modules stand on their seam - a flat ring, good adhesion, and every
overhang above it is under 10 degrees. The foot lies sole-down and has to
be rotated on the plate to fit its 243mm diagonally. The sandal is flat.

  1  paduka base        wood-fibre        + 3 ink bodies
  2  foot               ground filament   + 3 ink bodies
  3  shank              ground filament   + 3 ink bodies
  4  cuff               ground filament   + 3 ink bodies

Assembly is two 4mm pins and no tools.
"""

import zipfile
import numpy as np
import trimesh
import shapely

import paduka_stack as S
import paduka_modular as MOD
import paduka_sandal as SA
import paduka_pattern as PP
from paduka_3mf import CONTENT_TYPES, RELS, IDENT, _mesh_xml

INK_T = 1.50          # how deep the ink goes.
                      #
                      # This is NOT a cosmetic number, and 0.45 was wrong.
                      # On a vertical wall an inlay is only as wide as it is
                      # deep, and the slicer lays perimeters: at 0.45mm the
                      # pattern was thinner than a single 0.42mm extrusion,
                      # so the slicer dropped most of it and the rest came
                      # out as a dotted line. At 1.5mm every perimeter in
                      # the wall stack - two loops at 0.84mm, three at 1.26
                      # - falls inside the motif, and the motif prints
                      # solid. It is the wall thickness that sets this, not
                      # taste.
PRINT_MM = 1.30       # face size for the print meshes: the ink edge cannot
                      # be finer than the mesh that carries it
BED = 256.0

DYES = ("outline", "fill", "detail")
EXTRUDER = {"ground": 1, "outline": 2, "fill": 3, "detail": 4}


# ------------------------------------------------------------------- slabs ---

def _bnd(F, mask):
    """directed edges that bound the sub-shell F[mask], in mesh indices"""
    import trimesh.grouping as tg
    f = F[mask]
    if not len(f):
        return np.zeros((0, 2), int)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    g = tg.group_rows(np.sort(e, axis=1), require_count=1)
    return e[g] if len(g) else np.zeros((0, 2), int)


def _walls(e, n):
    """the quads that close a shell's rim onto its offset copy"""
    if not len(e):
        return []
    a, b = e[:, 0], e[:, 1]
    return [np.column_stack([a, b, b + n]),
            np.column_stack([a, b + n, a + n])]


def _solid(V, faces):
    m = trimesh.Trimesh(V, np.vstack(faces), process=False)
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    return m


def inlay(mesh, sels, t=INK_T):
    """
    Cut the pattern INTO the surface instead of standing it on top, and
    return the solids that result: the ground with recesses in it, and one
    plug per dye that fills its recess flush.

    Everything is built from a single sunk copy of the vertex array, so a
    plug's outer face IS the original surface and a recess floor IS that
    plug's back. They cannot drift apart, and there is no boolean to fail
    on a million-face mesh.

      ground   the un-inked faces, plus every inked face sunk by t as the
               recess floors, plus a wall round the rim of the whole
               inked region
      plug     one dye's faces at the surface, the same faces sunk as its
               back, and a wall round its own rim

    Inlay rather than relief, because the object is a printed CLOTH you
    wear. Block printing does leave a little relief on cotton, but on a leg
    you feel the pattern with a hand and catch it on a trouser leg, and a
    step round every motif is a dirt trap. Inlaid, the limb is smooth and
    the pattern is IN the material rather than on it - which is marquetry,
    and a better argument for a printed object anyway.

    The paduka keeps its relief. A sandal is a flat plate you stand on: the
    ornament costs nothing there, and proud ornament wears in rather than
    filling up.
    """
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    n = len(V)
    V2 = np.vstack([V, V - mesh.vertex_normals * t])

    ink = np.zeros(len(F), bool)
    for m in sels.values():
        ink |= m

    ground = _solid(V2, [F[~ink], F[ink] + n] + _walls(_bnd(F, ~ink), n))
    plugs = {}
    for k, m in sels.items():
        plugs[k] = _solid(V2, [F[m], (F[m] + n)[:, ::-1]]
                          + _walls(_bnd(F, m), n))
    return ground, plugs


def slab(shell, t=INK_T):
    """
    Lift an open shell of faces into a closed solid t thick, pushed OUT
    along the vertex normals. The paduka's relief still uses this.
    """
    import trimesh.grouping as tg
    v = np.asarray(shell.vertices, float)
    f = np.asarray(shell.faces, int)
    n = len(v)
    V = np.vstack([v, v + shell.vertex_normals * t])
    g = tg.group_rows(shell.edges_sorted, require_count=1)
    e = shell.edges[g] if len(g) else np.zeros((0, 2), int)
    return _solid(V, [f[:, ::-1], f + n] + _walls(e, n))


def print_bodies(way="madder", mods=None):
    """{module: {dye: mesh}} - the ground with its recesses, and the plugs"""
    mods = mods or S.modules()
    pat = S.drawing(way)
    polys = S._joint_polys()
    out = {}
    for nm, solid in mods.items():
        d = solid.subdivide_to_size(PRINT_MM)
        c = d.triangles.mean(axis=1)
        pts = shapely.points(S.face_uv(c))
        taken = S.buried(c, polys)
        sels = {}
        for k in DYES:
            g = pat.get(k)
            if g is None or g.is_empty:
                continue
            shapely.prepare(g)
            m = shapely.contains(g, pts) & ~taken
            if m.sum():
                sels[k] = m
            taken |= m
        ground, plugs = inlay(d, sels)
        out[nm] = dict(ground=ground, **plugs)
        del d, pts, sels
    S._PAT.clear()          # the drawing is tens of MB of shapely and the
                            # print meshes need every megabyte
    return out


def sandal_bodies():
    """the paduka: plate + rail + post, and the three ink blocks in relief"""
    T = SA.build_tree(shell=True)
    sole = T["sole"]
    plate = trimesh.util.concatenate(MOD._prism(sole, 0.0, MOD.PLATE_T))
    rail = MOD.dovetail(MOD.RAIL_W, MOD.RAIL_H, MOD.RAIL_X1 - MOD.RAIL_X0,
                        MOD.RAIL_X0, PP.centre_at(sole, 124.0))
    rail.apply_translation([0, 0, MOD.PLATE_T])
    base = trimesh.boolean.union([plate, rail, MOD.toe_post()],
                                 engine="manifold")
    out = {"ground": base}
    for k in DYES:
        parts = []
        for h, g in T["bodies"][k].items():
            if not g.is_empty:
                parts += MOD._prism(g, MOD.PLATE_T - 0.4, MOD.PLATE_T + h)
        if parts:
            out[k] = trimesh.util.concatenate(parts)
    return out


# -------------------------------------------------------------------- 3mf ---

def write_3mf(bodies, path, title, lay_flat=True):
    """one object, one part per dye, already assigned to filaments 1..4"""
    use = [(k, bodies[k]) for k in ("ground",) + DYES if k in bodies]
    ms = [m.copy() for _k, m in use]
    allv = np.vstack([np.asarray(m.vertices) for m in ms])
    lo, hi = allv.min(0), allv.max(0)
    off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for m in ms:
        m.apply_translation(off)

    objs, rows = [], []
    for i, ((k, _m), m) in enumerate(zip(use, ms), start=1):
        objs.append('<object id="%d" type="model" name="%s">%s</object>'
                    % (i, k, _mesh_xml(m)))
        rows.append('  <part id="%d" subtype="normal_part">\n'
                    '   <metadata key="name" value="%s"/>\n'
                    '   <metadata key="extruder" value="%d"/>\n'
                    '  </part>' % (i, k, EXTRUDER[k]))
    oid = len(ms) + 1
    comps = "".join('<component objectid="%d" transform="%s"/>' % (i, IDENT)
                    for i in range(1, len(ms) + 1))
    objs.append('  <object id="%d" type="model" name="%s"><components>%s'
                '</components></object>' % (oid, title, comps))
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/'
             '2015/02">\n <metadata name="Application">Paduka</metadata>\n'
             ' <metadata name="Title">%s</metadata>\n'
             ' <resources>\n%s\n </resources>\n'
             ' <build><item objectid="%d" transform="%s"/></build>\n'
             '</model>\n' % (title, "\n".join(objs), oid, IDENT))
    cfg = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
           ' <object id="%d">\n  <metadata key="name" value="%s"/>\n'
           '%s\n </object>\n</config>\n' % (oid, title, "\n".join(rows)))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/Slic3r_PE_model_config.xml", cfg)
        z.writestr("Metadata/model_settings.config", cfg)
    return ms


def report(name, ms):
    allv = np.vstack([np.asarray(m.vertices) for m in ms])
    e = allv.max(0) - allv.min(0)
    diag = np.hypot(e[0], e[1])
    fit = "fits" if max(e[0], e[1]) < BED - 6 else (
        "rotate 45 deg" if diag < (BED - 6) * 1.41 else "TOO BIG")
    # sum every body, not just the watertight ones: an inlaid ground has a
    # few non-manifold edges where two recess rims touch, which slicers
    # handle and which does not make its volume wrong
    vol = sum(abs(m.volume) for m in ms) / 1000
    print(f"  {name:12} {e[0]:6.1f} x {e[1]:5.1f} x {e[2]:6.1f} mm   "
          f"{sum(len(m.faces) for m in ms):8d} tris  {vol:6.1f} cm3   {fit}")


def hardware_bodies(n_bump=6, n_pin=2):
    """
    The small parts: bumpers and pins, on one plate, one filament.

    Six bumpers because they are the tuning and they cost 3cc each - print
    a set at 8%, 12% and 20% infill and walk on the one you like. Two pins
    because you need two and losing one stops the leg.

    The pins lie DOWN. A pin is loaded in bending across its axis, and a
    printed rod standing on end is a stack of discs held together by nothing
    but layer adhesion; lying down, the filament runs the length of it.
    """
    import paduka_ankle as AK
    out = []
    for i in range(n_bump):
        b = AK.bumper()
        b.apply_translation([(i % 3) * 24 - 24, (i // 3) * 28 - 14, 0])
        out.append(b)
    for i in range(n_pin):
        r = trimesh.creation.cylinder(radius=2.9, height=62, sections=48)
        r.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0]))
        r.apply_translation([i * 12 - 6, 0, 2.9])
        r.apply_translation([0, 46, 0])
        out.append(r)
    return {"ground": trimesh.util.concatenate(out)}


if __name__ == "__main__":
    import sys, time, gc
    way = sys.argv[1] if len(sys.argv) > 1 else "madder"
    only = set(sys.argv[2:])
    t0 = time.time()
    print(f"paduka print set - {way}")

    if not only or "paduka" in only:
        sb = sandal_bodies()
        report("paduka", write_3mf(sb, f"print_paduka_{way}.3mf",
                                   "Paduka sandal"))
        del sb; gc.collect()

    if not only or "hardware" in only:
        report("hardware", write_3mf(hardware_bodies(),
                                     f"print_hardware_{way}.3mf",
                                     "Paduka bumpers and pins"))

    mods = S.modules()
    for nm in (only & set(S.NAMES)) or S.NAMES:
        b = print_bodies(way, {nm: mods[nm]})[nm]
        report(nm, write_3mf(b, f"print_{nm}_{way}.3mf", f"Paduka {nm}"))
        del b; gc.collect()
    print(f"done in {time.time()-t0:.0f}s")
