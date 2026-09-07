#!/usr/bin/env python3
"""Shared style and map framing for the manuscript figures."""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
MM = 1.0 / 25.4
WIDTH_SINGLE = 89 * MM
WIDTH_DOUBLE = 183 * MM
OUTPUT_DIR = Path(os.environ.get('CONUSSM_FIGURE_OUTPUT', Path(__file__).resolve().parents[1] / 'output'))
NETWORK_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
OTHER_COLOR = '#9A9A98'
OTHER_LABEL = 'Other'
N_NAMED_NETWORKS = 6
SEQ_COLORS = ['#D3E4F5', '#A9C9E8', '#7BA9D9', '#3D7EBF', '#17539E']
NODATA_COLOR = '#FFFFFF'
NODATA_EDGE = '#CFCFCC'
DEPTH_COLORS = ['#FDB863', '#E66101', '#B2182B', '#762A83', '#2D004B']
LINE = 0.64
INK = '#2E2E2C'
MUTED = '#63635F'
GRID = '#DCDCD8'

def apply() -> None:
    """Install the manuscript rcParams. Call once at the top of a builder."""
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7.5, 'axes.titlesize': 8.0, 'axes.labelsize': 7.5, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.0, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': LINE, 'xtick.major.width': LINE, 'ytick.major.width': LINE, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'ps.fonttype': 42})

def panel_label(axis, text: str, x: float=-0.02, y: float=1.04) -> None:
    """Place a bold panel letter in axes coordinates."""
    axis.text(x, y, text, transform=axis.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='left', color=INK)

def network_palette(names: list[str]) -> dict[str, str]:
    """Map network names to fixed hues, folding the tail into OTHER."""
    palette = {name: NETWORK_COLORS[i] for i, name in enumerate(names[:N_NAMED_NETWORKS])}
    palette[OTHER_LABEL] = OTHER_COLOR
    return palette

def save(figure, name: str) -> None:
    """Write the PNG render used by the manuscript (400 dpi)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f'{name}.png'
    figure.savefig(target)
    print(f'wrote {target}')
    plt.close(figure)
CONUS_MERIDIANS = [-120, -105, -90, -75]
CONUS_PARALLELS = [30, 40]
MAP_LAND = '#F4F4F1'
MAP_STATES = '#D2D2CE'
MAP_COAST = '#A8A8A4'
CONUS_PROJ = dict(central_longitude=-96, standard_parallels=(29.5, 45.5))
CONUS_XLIM = (-2375106.0, 2276502.2)
CONUS_YLIM = (2615058.1, 5543790.5)
CONUS_ASPECT = (CONUS_XLIM[1] - CONUS_XLIM[0]) / (CONUS_YLIM[1] - CONUS_YLIM[0])

def conus_frame(axis, lon=None, lat=None, field=None, points=False, pad_deg=0.2, land=True, left_labels=True, bottom_labels=True, label_size=7.0):
    """Frame a CONUS map the way the figure set agreed to draw one."""
    import numpy as _np
    import cartopy.crs as _ccrs
    import cartopy.feature as _cf
    import matplotlib.ticker as _mt
    from fig_station_support import frame_ticks as _ft
    if land:
        axis.add_feature(_cf.LAND.with_scale('50m'), facecolor=MAP_LAND, zorder=0)
    axis.add_feature(_cf.STATES.with_scale('50m'), edgecolor=MAP_STATES, linewidth=0.22, zorder=1)
    axis.add_feature(_cf.COASTLINE.with_scale('50m'), edgecolor=MAP_COAST, linewidth=0.32, zorder=1)
    if lon is None or lat is None:
        axis.set_xlim(*CONUS_XLIM)
        axis.set_ylim(*CONUS_YLIM)
    else:
        lo, la = (_np.asarray(lon), _np.asarray(lat))
        if points:
            gx, gy = (lo, la)
        else:
            gx, gy = _np.meshgrid(lo, la)
            if field is not None:
                keep = _np.isfinite(_np.asarray(field, dtype=float))
                if keep.shape == gx.shape and keep.any():
                    gx, gy = (gx[keep], gy[keep])
        pts = axis.projection.transform_points(_ccrs.PlateCarree(), _np.ravel(gx), _np.ravel(gy))
        m = pad_deg * 111000.0
        axis.set_xlim(_np.nanmin(pts[:, 0]) - m, _np.nanmax(pts[:, 0]) + m)
        axis.set_ylim(_np.nanmin(pts[:, 1]) - m, _np.nanmax(pts[:, 1]) + m)
    g = axis.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.22, color='#E8E8E4')
    g.set_zorder(1.5)
    g.top_labels = g.right_labels = False
    g.left_labels, g.bottom_labels = (left_labels, bottom_labels)
    g.rotate_labels = False
    g.xlocator = _mt.FixedLocator(CONUS_MERIDIANS)
    g.ylocator = _mt.FixedLocator(CONUS_PARALLELS)
    g.xlabel_style = g.ylabel_style = {'size': label_size, 'color': INK}
    g.xpadding = g.ypadding = 6
    xt, yt = _ft(axis, CONUS_MERIDIANS, CONUS_PARALLELS)
    axis.xaxis.set_visible(True)
    axis.yaxis.set_visible(True)
    axis.set_xticks(xt if bottom_labels else [])
    axis.set_yticks(yt if left_labels else [])
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(axis='both', length=2.6, width=LINE, color=INK, top=False, right=False, labelbottom=False, labelleft=False)
    axis.spines['geo'].set_visible(True)
    axis.spines['geo'].set_linewidth(LINE)
    axis.spines['geo'].set_edgecolor(INK)
    return g
