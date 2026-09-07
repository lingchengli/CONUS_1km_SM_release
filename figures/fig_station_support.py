#!/usr/bin/env python3
"""Plot station locations and annual observation support by soil layer."""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import plot_style as st
DATA_DIR = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
AVAIL = DATA_DIR / 'station_availability.csv'
LAYERS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
LAYER_LABELS = ['0-5', '5-15', '15-30', '30-60', '60-100']
YEARS = list(range(2001, 2026))
CELL_BINS = [1, 25000, 50000, 80000, 130000, np.inf]
CELL_LABELS = ['<25k', '25-50k', '50-80k', '80-130k', '>130k']
SITE_BINS = [1, 2000, 8000, 20000, np.inf]
SITE_LABELS = ['<2k', '2-8k', '8-20k', '>20k']
SITE_SIZES = [6.0, 16.0, 34.0, 62.0]
LEGEND_FRAME = {'frameon': True, 'facecolor': 'white', 'edgecolor': st.GRID, 'framealpha': 0.92}

def load_cache() -> list[dict]:
    src = AVAIL
    if not src.exists():
        raise SystemExit(f'missing {src}; set CONUSSM_FIGURE_DATA to the derived-data directory')
    with src.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if row['lat'] == '' or row['lon'] == '':
            row['lat'] = row['lon'] = 'nan'
        row['lat'] = float(row['lat'])
        row['lon'] = float(row['lon'])
        row['year'] = int(row['year'])
        row['n_days_qc'] = int(row['n_days_total'])
        row['n_days_frozen'] = int(row['n_days_frozen'])
        out.append(row)
    return out

def aggregate(rows: list[dict]):
    site_days: dict[str, int] = defaultdict(int)
    site_meta: dict[str, dict] = {}
    network_sites: dict[str, set[str]] = defaultdict(set)
    grid = np.zeros((len(LAYERS), len(YEARS)), dtype=float)
    layer_sites: dict[str, set[str]] = defaultdict(set)
    layer_days: dict[str, int] = defaultdict(int)
    li = {layer: i for i, layer in enumerate(LAYERS)}
    yi = {year: i for i, year in enumerate(YEARS)}
    for row in rows:
        sid, layer, year, n = (row['site_id'], row['layer'], row['year'], row['n_days_qc'])
        site_days[sid] += n
        site_meta.setdefault(sid, {'network': row['network'], 'lat': row['lat'], 'lon': row['lon']})
        network_sites[row['network']].add(sid)
        layer_sites[layer].add(sid)
        layer_days[layer] += n
        if layer in li and year in yi:
            grid[li[layer], yi[year]] += n
    return (site_days, site_meta, network_sites, grid, layer_sites, layer_days)

def frame_ticks(axis, lons, lats):
    """Projected positions where labelled meridians and parallels meet the frame."""
    projection, source = (axis.projection, ccrs.PlateCarree())
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()

    def crossing(varying, constant, along_meridian, edge):
        if along_meridian:
            points = projection.transform_points(source, np.full_like(varying, constant), varying)
            track, other = (points[:, 1], points[:, 0])
        else:
            points = projection.transform_points(source, varying, np.full_like(varying, constant))
            track, other = (points[:, 0], points[:, 1])
        hits = np.flatnonzero(np.diff(np.sign(track - edge)))
        if hits.size == 0:
            return None
        i = hits[0]
        span = track[i + 1] - track[i]
        if span == 0:
            return other[i]
        return other[i] + (edge - track[i]) / span * (other[i + 1] - other[i])
    latitudes = np.linspace(15.0, 60.0, 4001)
    longitudes = np.linspace(-140.0, -50.0, 4001)
    x_ticks = [x for x in (crossing(latitudes, lon, True, y_min) for lon in lons) if x is not None and x_min <= x <= x_max]
    y_ticks = [y for y in (crossing(longitudes, lat, False, x_min) for lat in lats) if y is not None and y_min <= y <= y_max]
    return (x_ticks, y_ticks)

def panel_map(axis, site_days, site_meta, network_sites, palette, named):
    axis.set_extent([-124.8, -66.9, 23.5, 50.5], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#F4F4F1', zorder=0)
    axis.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#FFFFFF', zorder=0)
    axis.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='#C9C9C5', linewidth=0.35, zorder=1)
    axis.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#A8A8A4', linewidth=0.45, zorder=1)
    axis.spines['geo'].set_visible(True)
    axis.spines['geo'].set_linewidth(st.LINE)
    axis.spines['geo'].set_edgecolor(st.INK)
    graticule = axis.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.35, color=st.GRID, alpha=0.9, linestyle=(0, (2, 2)), zorder=2)
    meridians = [-120, -110, -100, -90, -80, -70]
    parallels = [25, 30, 35, 40, 45, 50]
    graticule.top_labels = False
    graticule.right_labels = False
    graticule.rotate_labels = False
    graticule.xlocator = mticker.FixedLocator(meridians)
    graticule.ylocator = mticker.FixedLocator(parallels)
    graticule.xlabel_style = {'size': 7.0, 'color': st.INK}
    graticule.ylabel_style = {'size': 7.0, 'color': st.INK}
    graticule.xpadding = 6
    graticule.ypadding = 6
    x_ticks, y_ticks = frame_ticks(axis, meridians, parallels)
    axis.xaxis.set_visible(True)
    axis.yaxis.set_visible(True)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(axis='both', which='major', length=2.5, width=st.LINE, color=st.INK, top=False, right=False, labelbottom=False, labelleft=False)
    size_class = np.digitize([site_days[s] for s in site_days], SITE_BINS[1:-1])
    sites = list(site_days)
    order = np.argsort(-np.array([site_days[s] for s in sites]))
    for index in order:
        sid = sites[index]
        meta = site_meta[sid]
        label = meta['network'] if meta['network'] in named else st.OTHER_LABEL
        axis.scatter(meta['lon'], meta['lat'], s=SITE_SIZES[size_class[index]], facecolor=palette[label], edgecolor='white', linewidth=0.25, alpha=0.6, transform=ccrs.PlateCarree(), zorder=3)
    hue_handles = [Line2D([], [], marker='o', linestyle='none', markersize=4, markerfacecolor=palette[name], markeredgecolor='white', markeredgewidth=0.3, label=f'{name} ({len(network_sites[name])})') for name in named]
    other_n = sum((len(v) for k, v in network_sites.items() if k not in named))
    hue_handles.append(Line2D([], [], marker='o', linestyle='none', markersize=4, markerfacecolor=st.OTHER_COLOR, markeredgecolor='white', markeredgewidth=0.3, label=f'{st.OTHER_LABEL} ({other_n})'))
    first = axis.legend(handles=hue_handles, loc='lower left', bbox_to_anchor=(0.01, 0.012), ncol=2, handletextpad=0.3, columnspacing=0.7, labelspacing=0.3, borderaxespad=0.0, borderpad=0.45, fontsize=6.8, title='Network (sites)', title_fontsize=7.0, **LEGEND_FRAME)
    first._legend_box.align = 'left'
    axis.add_artist(first)
    size_handles = [Line2D([], [], marker='o', linestyle='none', markersize=np.sqrt(size), markerfacecolor='#7A7A78', markeredgecolor='white', markeredgewidth=0.3, label=label) for size, label in zip(SITE_SIZES, SITE_LABELS)]
    second = axis.legend(handles=size_handles, loc='lower right', bbox_to_anchor=(0.99, 0.012), ncol=1, handletextpad=0.5, labelspacing=0.42, borderaxespad=0.0, borderpad=0.45, fontsize=6.8, title='Station-days per site', title_fontsize=7.0, **LEGEND_FRAME)
    second._legend_box.align = 'left'
    for legend in (first, second):
        legend.get_frame().set_linewidth(st.LINE)

def panel_grid(axis, grid):
    cmap = ListedColormap(st.SEQ_COLORS)
    cmap.set_bad(st.NODATA_COLOR)
    norm = BoundaryNorm(CELL_BINS, cmap.N)
    masked = np.ma.masked_where(grid <= 0, grid)
    axis.pcolormesh(np.arange(len(YEARS) + 1), np.arange(len(LAYERS) + 1), masked, cmap=cmap, norm=norm, edgecolors='white', linewidth=0.7)
    axis.set_ylim(len(LAYERS), 0)
    axis.set_aspect('equal', adjustable='box', anchor='NW')
    for r in range(len(LAYERS)):
        for c in range(len(YEARS)):
            if grid[r, c] <= 0:
                axis.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=st.NODATA_COLOR, edgecolor=st.NODATA_EDGE, linewidth=0.5, zorder=3))
    labelled = [y for y in YEARS if y % 4 == 1 or y == YEARS[-1]]
    axis.set_xticks([YEARS.index(y) + 0.5 for y in labelled])
    axis.set_xticklabels([str(y) for y in labelled])
    axis.set_xticks(np.arange(len(YEARS)) + 0.5, minor=True)
    axis.set_yticks(np.arange(len(LAYERS)) + 0.5)
    axis.set_yticklabels(LAYER_LABELS)
    axis.set_ylabel('Layer (cm)')
    axis.tick_params(axis='both', which='major', length=3.0, width=st.LINE)
    axis.tick_params(axis='x', which='minor', length=1.5, width=0.5)
    for side in ('bottom', 'left'):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(st.LINE)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)

def panel_bars(axis, values, color, label, fmt, ticks, tick_labels):
    y = np.arange(len(LAYERS)) + 0.5
    axis.barh(y, values, height=0.34, color=color, edgecolor='none')
    axis.set_ylim(len(LAYERS), 0)
    axis.set_xlim(0, max(values) * 1.46)
    for yi, value in zip(y, values):
        axis.text(value * 1.06, yi, fmt(value), va='center', ha='left', fontsize=6.5, color=st.INK)
    axis.set_yticks([])
    axis.set_xticks(ticks)
    axis.set_xticklabels(tick_labels, fontsize=6.5)
    axis.set_xlabel(label, fontsize=6.8, labelpad=2)
    axis.tick_params(axis='x', length=2.5, width=st.LINE)
    axis.spines['bottom'].set_visible(True)
    axis.spines['bottom'].set_linewidth(st.LINE)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', action='store_true', help='print distributions used to choose the bin breaks')
    args = parser.parse_args()
    st.apply()
    rows = load_cache()
    site_days, site_meta, network_sites, grid, layer_sites, layer_days = aggregate(rows)
    named = sorted(network_sites, key=lambda k: -len(network_sites[k]))[:st.N_NAMED_NETWORKS]
    palette = st.network_palette(named)
    site_counts = [len(layer_sites[layer]) for layer in LAYERS]
    day_totals = [layer_days[layer] for layer in LAYERS]
    expect = 1041
    if len(site_meta) != expect:
        raise RuntimeError(f'target-stage site union {len(site_meta)} != {expect}')
    if sum(day_totals) != sum((r['n_days_qc'] for r in rows)):
        raise RuntimeError('layer station-day totals do not reconcile with the cache')
    if args.diagnostics:
        per_site = np.array(sorted(site_days.values()))
        print('per-site station-days percentiles:', {p: int(np.percentile(per_site, p)) for p in (5, 25, 50, 75, 95, 99)})
        print('max cell station-days:', int(grid.max()))
        print('layer site counts:', dict(zip(LAYER_LABELS, site_counts)))
        print('layer station-days:', dict(zip(LAYER_LABELS, day_totals)))
        print('networks:', {k: len(v) for k, v in sorted(network_sites.items(), key=lambda kv: -len(kv[1]))})
    figure = plt.figure(figsize=(st.WIDTH_DOUBLE, 6.2))
    outer = figure.add_gridspec(2, 1, height_ratios=[4.6, 1.4], hspace=0.3)
    map_axis = figure.add_subplot(outer[0], projection=ccrs.AlbersEqualArea(central_longitude=-96.0, central_latitude=23.0, standard_parallels=(29.5, 45.5)))
    panel_map(map_axis, site_days, site_meta, network_sites, palette, named)
    st.panel_label(map_axis, 'a', x=0.0, y=1.01)
    lower = outer[1].subgridspec(1, 3, width_ratios=[7.4, 1.05, 1.05], wspace=0.16)
    grid_axis = figure.add_subplot(lower[0])
    panel_grid(grid_axis, grid)
    st.panel_label(grid_axis, 'b', x=0.0, y=1.06)
    site_axis = figure.add_subplot(lower[1])
    days_axis = figure.add_subplot(lower[2])
    panel_bars(site_axis, site_counts, '#7A7A78', 'Sites', lambda v: f'{int(v):,}', ticks=[0, 1000], tick_labels=['0', '1,000'])
    panel_bars(days_axis, day_totals, '#3D7EBF', 'Station-days', lambda v: f'{v / 1000.0:,.0f}k', ticks=[0, 2000000], tick_labels=['0', '2M'])
    figure.canvas.draw()
    box = grid_axis.get_position()
    for bar_axis in (site_axis, days_axis):
        pos = bar_axis.get_position()
        bar_axis.set_position([pos.x0, box.y0, pos.width, box.height])
    figure_w, figure_h = figure.get_size_inches()
    cell_w = box.width * figure_w / len(YEARS)
    cell_h = box.height * figure_h / len(LAYERS)
    if abs(cell_w - cell_h) > 0.0005:
        raise RuntimeError(f'layer-year cells are not square: {cell_w:.4f} x {cell_h:.4f} in; give the lower gridspec row more height')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='none') for c in st.SEQ_COLORS]
    labels = list(CELL_LABELS)
    if (grid <= 0).any():
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=st.NODATA_COLOR, edgecolor=st.NODATA_EDGE, linewidth=0.4))
        labels.append('no data')
    grid_axis.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=len(labels), handlelength=1.3, handleheight=0.85, handletextpad=0.4, columnspacing=1.0)
    st.save(figure, 'station_support')
    return 0
if __name__ == '__main__':
    sys.exit(main())
