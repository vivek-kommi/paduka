"""Minimal 2D preview helper: draw a shapely geometry onto a matplotlib axis."""
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def _ring_codes(n):
    return [Path.MOVETO] + [Path.LINETO] * (n - 2) + [Path.CLOSEPOLY]


def draw(ax, geom, fc="#000000", ec="none", lw=0.6, zorder=1, alpha=1.0,
         **kw):
    if geom is None or geom.is_empty:
        return
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    verts, codes = [], []
    for p in polys:
        if p.geom_type != "Polygon":
            continue
        for ring in [p.exterior] + list(p.interiors):
            c = np.asarray(ring.coords)
            if len(c) < 3:
                continue
            verts.append(c)
            codes += _ring_codes(len(c))
    if not verts:
        return
    path = Path(np.vstack(verts), codes)
    ax.add_patch(PathPatch(path, facecolor=fc, edgecolor=ec, lw=lw,
                           zorder=zorder, alpha=alpha, **kw))
