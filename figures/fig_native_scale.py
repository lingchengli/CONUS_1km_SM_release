#!/usr/bin/env python3
"""Compare monthly spatial structure at each product's native grid spacing."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as st
from fig_station_support import frame_ticks
DATA_DIR = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
ROWS = [('ours', 'ConusSM1k (1/120°)', ''), ('swsm', 'SWSM (0.05°)', ''), ('era5', 'ERA5-Land (0.1°)', ''), ('gleam', 'GLEAM (0.1°)', ''), ('cci', 'ESA CCI (0.25°)', '')]
WINDOW_NAMES = ['Central Valley, CA', 'Front Range, CO', 'East Texas', 'Corn Belt, IA', 'Delmarva, MD-VA']
GLOW = [patheffects.withStroke(linewidth=2.6, foreground='white')]
NW = len(WINDOW_NAMES)
DIVERGING = ['#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#F5F5F5', '#C7EAE5', '#80CDC1', '#35978F', '#01665E']
CLIP_PCT = 98.0
LABEL_X = 0.128
PAD = 0.2
MERIDIANS = [-120, -105, -90, -75]
PARALLELS = [30, 40]

def main() -> int:
    data = np.load(DATA_DIR / 'native_scale.npz')
    month = str(data['month'])
    windows = []
    for index in range(NW):
        lat = data[f'ours_w{index}_lat']
        lon = data[f'ours_w{index}_lon']
        dlat = float(abs(lat[1] - lat[0]))
        dlon = float(abs(lon[1] - lon[0]))
        windows.append({'lat0': float(np.min(lat) - dlat / 2), 'lat1': float(np.max(lat) + dlat / 2), 'lon0': float(np.min(lon) - dlon / 2), 'lon1': float(np.max(lon) + dlon / 2)})
    label = f'{month[:4]}-{month[4:]}'
    loc = data['ours_locator']
    llat, llon = (data['ours_locator_lat'], data['ours_locator_lon'])
    rows, cols = np.nonzero(np.isfinite(loc))
    extent = (float(llon[cols.min()]) - PAD, float(llon[cols.max()]) + PAD, float(llat[rows.max()]) - PAD, float(llat[rows.min()]) + PAD)
    pooled = np.concatenate([np.abs(data[f'{k}_w{i}'][np.isfinite(data[f'{k}_w{i}'])]) for k, _, _ in ROWS for i in range(NW)])
    vmax = float(np.percentile(pooled, CLIP_PCT))
    cmap = LinearSegmentedColormap.from_list('brtl', DIVERGING)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    st.apply()
    proj = ccrs.AlbersEqualArea(central_longitude=-96, standard_parallels=(29.5, 45.5))
    fin0 = np.isfinite(data['ours_locator'])
    gy0, gx0 = np.nonzero(fin0[::4, ::4])
    pp = proj.transform_points(ccrs.PlateCarree(), data['ours_locator_lon'][::4][gx0], data['ours_locator_lat'][::4][gy0])
    m_in = PAD * 111000.0
    aspect = (pp[:, 0].max() - pp[:, 0].min() + 2 * m_in) / (pp[:, 1].max() - pp[:, 1].min() + 2 * m_in)
    LEFT, RIGHT = (0.145, 0.995)
    WSP = 0.06
    W = st.WIDTH_DOUBLE
    cell_f = (RIGHT - LEFT) / (NW + WSP * (NW - 1))
    MAP_RIGHT = LEFT + 3 * cell_f + 2 * WSP * cell_f
    map_w = (MAP_RIGHT - LEFT) * W
    map_h = map_w / aspect
    cell_w = cell_f * W
    NR = len(ROWS)
    grid_h = NR * cell_w + (NR - 1) * 0.14 * cell_w
    m_top, m_gap, m_bot = (0.28, 0.44, 0.1)
    fig_h = m_top + map_h + m_gap + grid_h + m_bot
    fig = plt.figure(figsize=(W, fig_h))
    top = fig.add_gridspec(1, 1, left=LEFT, right=MAP_RIGHT, top=1 - m_top / fig_h, bottom=1 - (m_top + map_h) / fig_h)
    grid = fig.add_gridspec(NR, NW, left=LEFT, right=RIGHT, top=(grid_h + m_bot) / fig_h, bottom=m_bot / fig_h, wspace=WSP, hspace=0.14)
    axis = fig.add_subplot(top[0, 0], projection=proj)
    fin = np.isfinite(loc)
    gy, gx = np.nonzero(fin[::4, ::4])
    pts = proj.transform_points(ccrs.PlateCarree(), llon[::4][gx], llat[::4][gy])
    px, py = (pts[:, 0], pts[:, 1])
    mx = PAD * 111000.0
    axis.set_xlim(px.min() - mx, px.max() + mx)
    axis.set_ylim(py.min() - mx, py.max() + mx)
    axis.pcolormesh(data['ours_locator_lon'], data['ours_locator_lat'], np.ma.masked_invalid(data['ours_locator']), cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
    axis.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#9A9A96', linewidth=0.22, zorder=2)
    for i, w in enumerate(windows):
        axis.add_patch(Rectangle((w['lon0'], w['lat0']), w['lon1'] - w['lon0'], w['lat1'] - w['lat0'], transform=ccrs.PlateCarree(), facecolor='none', edgecolor=st.INK, linewidth=1.0, zorder=4))
        a = proj.transform_points(ccrs.PlateCarree(), np.array([w['lon0'], w['lon0']]), np.array([w['lat0'], w['lat1']]))
        angle = np.degrees(np.arctan2(a[1, 1] - a[0, 1], a[1, 0] - a[0, 0])) - 90.0
        axis.text(w['lon0'] + 0.32, w['lat1'] - 0.3, str(i + 1), transform=ccrs.PlateCarree(), fontsize=7.4, fontweight='bold', color=st.INK, ha='left', va='top', zorder=5, rotation=angle, rotation_mode='anchor', path_effects=GLOW, clip_on=False)
    g = axis.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.22, color='#E8E8E4')
    g.set_zorder(0.5)
    g.top_labels = g.right_labels = False
    g.rotate_labels = False
    g.xlocator = mticker.FixedLocator(MERIDIANS)
    g.ylocator = mticker.FixedLocator(PARALLELS)
    g.xlabel_style = g.ylabel_style = {'size': 7.0, 'color': st.INK}
    g.xpadding = g.ypadding = 6
    xt, yt = frame_ticks(axis, MERIDIANS, PARALLELS)
    axis.xaxis.set_visible(True)
    axis.yaxis.set_visible(True)
    axis.set_xticks(xt)
    axis.set_yticks(yt)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(axis='both', length=2.6, width=st.LINE, color=st.INK, top=False, right=False, labelbottom=False, labelleft=False)
    axis.spines['geo'].set_edgecolor(st.INK)
    axis.spines['geo'].set_linewidth(st.LINE)
    axis.set_title(f'ConusSM1k, monthly mean {label}', fontsize=8.0, pad=4)
    fig.text(LABEL_X, 1 - (m_top - 0.1) / fig_h, 'a', fontsize=8.5, fontweight='bold', va='bottom', ha='left', color=st.INK)
    row_axes = []
    for r, (key, name, res) in enumerate(ROWS):
        for c in range(NW):
            ax = fig.add_subplot(grid[r, c], projection=ccrs.PlateCarree())
            field = data[f'{key}_w{c}']
            lat, lon = (data[f'{key}_w{c}_lat'], data[f'{key}_w{c}_lon'])
            mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), cmap=cmap, norm=norm, rasterized=True, shading='nearest', transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), edgecolor='#55554F', linewidth=0.35, zorder=3)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_linewidth(st.LINE)
            ax.spines['geo'].set_edgecolor(st.INK)
            if r == 0:
                ax.set_title(f'{c + 1}. {WINDOW_NAMES[c]}', fontsize=7.4, pad=4)
            if c == 0:
                row_axes.append((ax, name))
                if r == 0:
                    fig.text(LABEL_X, (grid_h + m_bot + 0.1) / fig_h, 'b', fontsize=8.5, fontweight='bold', va='bottom', ha='left', color=st.INK)
            ax.text(0.975, 0.03, f'{field.shape[0]}×{field.shape[1]}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.0, color=st.INK, bbox=dict(boxstyle='square,pad=0.16', facecolor='white', edgecolor='none', alpha=0.8))
    cax = fig.add_axes([MAP_RIGHT + 0.035, 1 - (m_top + 0.8 * map_h) / fig_h, 0.016, 0.62 * map_h / fig_h])
    bar = fig.colorbar(mesh, cax=cax, extend='both')
    bar.set_label('relative to own domain mean (m$^3$ m$^{-3}$)', fontsize=7.6, labelpad=5)
    bar.ax.tick_params(labelsize=7.2, length=2.4, width=st.LINE)
    bar.outline.set_linewidth(st.LINE)
    bar.outline.set_edgecolor(st.INK)
    fig.canvas.draw()
    for ax, text in row_axes:
        box = ax.get_position()
        fig.text(LEFT - 0.014, 0.5 * (box.y0 + box.y1), text, rotation=90, fontsize=7.4, ha='right', va='center', color=st.INK, linespacing=1.4)
    pa = axis.get_position()
    c0 = fig.axes[1].get_position()
    c2 = fig.axes[3].get_position()
    print(f'  panel a   left={pa.x0:.4f} right={pa.x1:.4f}')
    print(f'  b col 1   left={c0.x0:.4f}')
    print(f'  b col 3   right={c2.x1:.4f}')
    for name, got, want in (('left', pa.x0, c0.x0), ('right', pa.x1, c2.x1)):
        if abs(got - want) > 0.002:
            raise RuntimeError(f'panel a {name} edge {got:.4f} != b {want:.4f}')
    st.save(fig, 'native_scale')
    return 0
if __name__ == '__main__':
    sys.exit(main())
