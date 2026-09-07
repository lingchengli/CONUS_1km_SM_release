#!/usr/bin/env python3
"""Plot surface-soil-moisture skill against common in-situ observations."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as st
from fig_station_support import frame_ticks
DATA_DIR = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
LAYER = '0-5cm'
ORDER = ['ConusSM1k', 'SWSM', 'ERA5-Land', 'GLEAM', 'ESA CCI']
COLOR_INDEX = {'ConusSM1k': 0, 'ERA5-Land': 1, 'SWSM': 2, 'GLEAM': 3, 'ESA CCI': 4}
COLOR = {product: st.NETWORK_COLORS[i] for product, i in COLOR_INDEX.items()}
if set(COLOR_INDEX) != set(ORDER):
    raise RuntimeError(f'COLOR_INDEX and ORDER disagree: {set(COLOR_INDEX) ^ set(ORDER)}')
RAMP = ['#e6f5b2', '#a0dab8', '#46b8c3', '#1e83ba', '#24439b', '#081d58']
EXTENT = (-125.0, -66.5, 24.0, 49.5)
MERIDIANS = [-120, -105, -90, -75]
PARALLELS = [30, 40]
N_BOOT = 2000
MAP_ASPECT = 1.586
R_RANGE = (0.0, 0.9)
METRICS = [('R_anom', 'anomaly R', None), ('ubRMSE', 'ubRMSE (m$^3$ m$^{-3}$)', None), ('bias', 'bias (m$^3$ m$^{-3}$)', 0.0)]
HALO = [patheffects.withStroke(linewidth=1.6, foreground='white')]
N_BLOCKS_EXPECTED = 25

def block_ci(values: np.ndarray, blocks: np.ndarray, rng) -> tuple[float, float]:
    """95% interval by resampling the KMeans BLOCKS, not the stations."""
    if np.isnan(np.asarray(blocks, dtype=float)).any():
        raise RuntimeError('unassigned spatial block reached the bootstrap')
    uniq = np.unique(blocks)
    if uniq.size != N_BLOCKS_EXPECTED:
        raise RuntimeError(f'expected {N_BLOCKS_EXPECTED} spatial blocks, got {uniq.size}')
    idx = {b: np.flatnonzero(blocks == b) for b in uniq}
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    med = np.empty(N_BOOT)
    for i in range(N_BOOT):
        med[i] = np.median(values[np.concatenate([idx[uniq[j]] for j in draws[i]])])
    return (float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5)))

def frame(axis) -> None:
    """Black x and y axis lines at the shared manuscript weight, and no grid."""
    for side in ('left', 'bottom'):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(st.LINE)
        axis.spines[side].set_color(st.INK)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)

def main() -> int:
    table = pd.read_csv(DATA_DIR / 'multiproduct_scores.csv')
    table = table[table.layer == LAYER]
    coords = pd.read_csv(DATA_DIR / 'station_summary.csv').groupby('site_id')[['lat', 'lon']].first()
    blocks = pd.read_csv(DATA_DIR / 'station_blocks.csv')
    blocks = blocks[blocks.layer == LAYER].set_index('site_id')['block']
    table = table.join(coords, on='site_id').dropna(subset=['lat', 'lon'])
    table['block'] = table['site_id'].map(blocks)
    rng = np.random.default_rng(33)
    st.apply()
    cmap = LinearSegmentedColormap.from_list('ramp', RAMP)
    norm = Normalize(*R_RANGE)
    proj = ccrs.AlbersEqualArea(central_longitude=-96, standard_parallels=(29.5, 45.5))
    W = st.WIDTH_DOUBLE
    LEFT, RIGHT, WSP, HSP = (0.055, 0.985, 0.05, 0.16)
    cell_w = (RIGHT - LEFT) * W / (3 + 2 * WSP)
    map_h = cell_w / MAP_ASPECT
    upper_h = 2 * map_h + HSP * map_h
    lower_h, m_top, m_mid, m_bot = (1.7, 0.299, 0.467, 0.38)
    fig_h = m_top + upper_h + m_mid + lower_h + m_bot
    fig = plt.figure(figsize=(W, fig_h))
    upper = fig.add_gridspec(2, 3, left=LEFT, right=RIGHT, top=1 - m_top / fig_h, bottom=1 - (m_top + upper_h) / fig_h, wspace=WSP, hspace=HSP)
    lower = fig.add_gridspec(1, 3, left=LEFT, right=RIGHT, top=(m_bot + lower_h) / fig_h, bottom=m_bot / fig_h, wspace=WSP)
    for k, product in enumerate(ORDER):
        row, col = divmod(k, 3)
        sub = table[table['product'] == product]
        axis = fig.add_subplot(upper[row, col], projection=proj)
        st.conus_frame(axis, left_labels=col == 0, bottom_labels=row == 1 or col == 2)
        order = np.argsort(-sub['R_anom'].to_numpy())
        axis.scatter(sub['lon'].to_numpy()[order], sub['lat'].to_numpy()[order], s=5.0, c=sub['R_anom'].to_numpy()[order], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), linewidths=0.14, edgecolors='#55554F', zorder=3)
        axis.set_title(f"{product}  ({sub['support'].iloc[0]})", fontsize=8.0, pad=4)
        axis.text(0.015, 0.04, f"median R$_a$ = {sub['R_anom'].median():.2f}", transform=axis.transAxes, ha='left', va='bottom', fontsize=7.2, color=st.INK, path_effects=HALO, zorder=4)
        if k == 0:
            st.panel_label(axis, 'a', x=-0.01, y=1.07)
    cell = upper[1, 2].get_position(fig)
    cax = fig.add_axes([cell.x0 + 0.3 * cell.width, cell.y0 + 0.16 * cell.height, 0.048 * cell.width, 0.68 * cell.height])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', extend='both')
    bar.set_label('anomaly R', fontsize=8.0, labelpad=5)
    bar.ax.tick_params(labelsize=7.2, length=2.4, width=st.LINE)
    bar.outline.set_linewidth(st.LINE)
    bar.outline.set_edgecolor(st.INK)
    present = [p for p in ORDER if p in set(table['product'])]
    ypos = np.arange(len(present))[::-1]
    ours = table[table['product'] == 'ConusSM1k'].set_index('site_id')
    paired = {}
    for product in present:
        if product == 'ConusSM1k':
            continue
        comp = table[table['product'] == product].set_index('site_id')
        common = ours.index.intersection(comp.index)
        diff = (ours.loc[common, 'R_anom'] - comp.loc[common, 'R_anom']).to_numpy()
        lo, hi = block_ci(diff, ours.loc[common, 'block'].to_numpy(), rng)
        paired[product] = (float(np.median(diff)), lo, hi)
    failures = {p: values for p, values in paired.items() if values[1] <= 0 <= values[2]}
    if failures:
        raise RuntimeError(f'caption states that every paired 95% block-bootstrap interval excludes zero, but these do not: {failures}')
    for j, (metric, label, ref) in enumerate(METRICS):
        axis = fig.add_subplot(lower[0, j])
        axis.set_box_aspect(0.8)
        st.panel_label(axis, 'bcd'[j], x=-0.02, y=1.03)
        data = [table[table['product'] == p][metric].to_numpy() for p in present]
        parts = axis.violinplot(data, positions=ypos, widths=0.5, orientation='horizontal', showextrema=False, showmedians=False)
        for body, p in zip(parts['bodies'], present):
            body.set_facecolor(COLOR[p])
            body.set_edgecolor('none')
            body.set_alpha(0.85)
        axis.boxplot(data, positions=ypos, widths=0.085, vert=False, showfliers=False, showcaps=False, whis=0, patch_artist=True, boxprops=dict(facecolor='white', edgecolor=st.INK, linewidth=0.5), medianprops=dict(color=st.INK, linewidth=1.1), whiskerprops=dict(linewidth=0))
        if ref is not None:
            axis.axvline(ref, color=st.MUTED, linewidth=0.5, linestyle=(0, (3, 2)))
        pooled = np.concatenate(data)
        span = np.percentile(pooled, [0.5, 99.5])
        pad = 0.06 * (span[1] - span[0])
        axis.set_xlim(span[0] - pad, span[1] + pad)
        axis.set_yticks(ypos)
        axis.set_ylim(-0.75, len(present) - 0.25)
        axis.set_yticklabels([])
        axis.set_xlabel(label, fontsize=8.0)
        axis.tick_params(axis='x', labelsize=7.2, width=st.LINE, length=2.4)
        axis.tick_params(axis='y', length=0)
        frame(axis)
        if metric == 'R_anom':
            x0, x1 = axis.get_xlim()
            for yi, p in zip(ypos, present):
                axis.text(x0 + 0.02 * (x1 - x0), yi + 0.29, p, va='bottom', ha='left', fontsize=8.0, color=st.INK, path_effects=HALO, zorder=5)
    st.save(fig, 'surface_comparison')
    return 0
if __name__ == '__main__':
    sys.exit(main())
