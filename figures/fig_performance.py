#!/usr/bin/env python3
"""Plot predicted-versus-observed densities and marginal distributions."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
import plot_style as st
HALO = [patheffects.withStroke(linewidth=1.8, foreground='white')]
DATA_DIR = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
LAYERS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
LABELS = ['0-5 cm', '5-15 cm', '15-30 cm', '30-60 cm', '60-100 cm']
OBS_INK = st.INK
ONE_TO_ONE = st.INK
FRAME = st.INK
LINE = st.LINE
SMOOTH_BINS = 1.6
NY_HEX = 12
NX_HEX = int(round(NY_HEX * np.sqrt(3)))
MARGIN_LEFT = 0.55
MARGIN_RIGHT = 0.06
MARGIN_TOP = 0.26
MARGIN_TOP_BANNER = 0.44
MARGIN_BOTTOM = 0.42
WIDTH_RATIOS = [1, 1, 1, 1, 1, 0.075]
HEIGHT_RATIOS = [1.0, 1.0]
WSPACE, HSPACE = (0.16, 0.24)
STOPS = (0.0, 0.22, 0.45, 0.65, 0.82, 1.0)
DENSITY_COLORS = ['#ffffd9', '#d0edb3', '#59bfc0', '#1e86bb', '#24479d', '#081d58']
DENSITY_ALPHA = [0.1, 0.26, 0.46, 0.68, 0.88, 1.0]
VMIN_HEX = 100
PROTOCOLS = {'spatial': {'out': 'performance_spatial', 'line': '#0072B2', 'banner': None}, 'temporal': {'out': 'performance_temporal', 'line': '#2E7D5B', 'banner': None}, 'rowrandom': {'out': 'performance_random_record', 'line': '#A9601F', 'banner': None}}

def load(tag):
    npz = DATA_DIR / f'{tag}_density.npz'
    js = DATA_DIR / f'{tag}_metrics.json'
    if not npz.exists() or not js.exists():
        raise SystemExit(f"missing cache for '{tag}': {npz.name}, {js.name}")
    return (np.load(npz), json.loads(js.read_text()))

def solve_height(banner: bool) -> tuple[float, float, float]:
    """Figure height that makes the joint panels exactly square, and the margins."""
    ncols, nrows = (len(WIDTH_RATIOS), len(HEIGHT_RATIOS))
    axes_width = st.WIDTH_DOUBLE - MARGIN_LEFT - MARGIN_RIGHT
    cell_w = axes_width / (ncols + WSPACE * (ncols - 1))
    panel_w = cell_w * ncols / sum(WIDTH_RATIOS) * WIDTH_RATIOS[0]
    share = HEIGHT_RATIOS[0] * nrows / ((nrows + HSPACE * (nrows - 1)) * sum(HEIGHT_RATIOS))
    axes_height = panel_w / share
    margin_top = MARGIN_TOP_BANNER if banner else MARGIN_TOP
    return (axes_height + margin_top + MARGIN_BOTTOM, margin_top, panel_w)

def smooth(density: np.ndarray) -> np.ndarray:
    """Binned KDE of a histogram-derived density. See SMOOTH_BINS."""
    return gaussian_filter1d(density, SMOOTH_BINS, mode='reflect')

def suppression_stats(counts: np.ndarray, expected_total: float) -> dict[str, float]:
    """Measure the display cut on the hexagons that Matplotlib actually drew."""
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or not values.size or (not np.isfinite(values).all()):
        raise RuntimeError('hexbin returned an empty or non-finite count array')
    if not np.isclose(values.sum(), expected_total, rtol=0, atol=0.5):
        raise RuntimeError(f'hexbin count is not conservative: {values.sum():.0f} != {expected_total:.0f} station-days')
    removed = values < VMIN_HEX
    return {'occupied': float(values.size), 'removed': float(removed.sum()), 'hex_pct': float(100.0 * removed.mean()), 'station_day_pct': float(100.0 * values[removed].sum() / values.sum())}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True, choices=sorted(PROTOCOLS))
    ap.add_argument('--free-y', action='store_true', help='let each marginal panel set its own density range')
    args = ap.parse_args()
    spec = PROTOCOLS[args.protocol]
    st.apply()
    data, report = load(args.protocol)
    lo, hi = [float(v) for v in data['rng']]
    edges = np.linspace(lo, hi, int(data['bins'][0]) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    cmap = LinearSegmentedColormap.from_list('density', list(zip(STOPS, [to_rgba(c, a) for c, a in zip(DENSITY_COLORS, DENSITY_ALPHA)])))
    density_max = 0.0
    for layer in LAYERS:
        hist = data[f'H__{layer}'].astype(float)
        total = hist.sum()
        density_max = max(density_max, smooth(hist.sum(1) / (total * width)).max(), smooth(hist.sum(0) / (total * width)).max())
    figure_h, margin_top, panel_w = solve_height(bool(spec['banner']))
    figure = plt.figure(figsize=(st.WIDTH_DOUBLE, figure_h))
    grid = figure.add_gridspec(len(HEIGHT_RATIOS), len(WIDTH_RATIOS), width_ratios=WIDTH_RATIOS, height_ratios=HEIGHT_RATIOS, wspace=WSPACE, hspace=HSPACE, left=MARGIN_LEFT / st.WIDTH_DOUBLE, right=1.0 - MARGIN_RIGHT / st.WIDTH_DOUBLE, top=1.0 - margin_top / figure_h, bottom=MARGIN_BOTTOM / figure_h)
    first_top = first_bottom = None
    hexes: list = []
    suppression: list[tuple[str, dict[str, float]]] = []
    for index, (layer, label) in enumerate(zip(LAYERS, LABELS)):
        pooled = report[layer]['pooled']
        hist = data[f'H__{layer}'].astype(float)
        total = hist.sum()
        axis = figure.add_subplot(grid[0, index])
        if index == 0:
            first_top = axis
        row, col = np.nonzero(hist)
        hexed = axis.hexbin(centres[row], centres[col], C=hist[row, col], reduce_C_function=np.sum, gridsize=(NX_HEX, NY_HEX), extent=(lo, hi, lo, hi), cmap=cmap, mincnt=1, linewidths=0.0, zorder=0.5, rasterized=True)
        hexes.append(hexed)
        suppression.append((label, suppression_stats(hexed.get_array(), expected_total=total)))
        axis.plot([lo, hi], [lo, hi], color=ONE_TO_ONE, lw=LINE, zorder=3, label='1:1')
        slope = pooled['rma_slope']
        mx = float(np.sum(hist.sum(1) * centres) / total)
        my = float(np.sum(hist.sum(0) * centres) / total)
        axis.plot([lo, hi], [my + slope * (lo - mx), my + slope * (hi - mx)], color=spec['line'], lw=LINE, ls=(0, (3.2, 1.6)), zorder=4, label='RMA fit')
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_aspect('equal', adjustable='box', anchor='S')
        axis.set_xticks([0, 0.2, 0.4, 0.6])
        axis.set_yticks([0, 0.2, 0.4, 0.6])
        axis.tick_params(labelbottom=False, length=2.6, width=LINE, pad=1.8)
        axis.set_title(label, fontsize=8.0, pad=3.5)
        for side in axis.spines.values():
            side.set_linewidth(LINE)
            side.set_color(FRAME)
        if index == 0:
            axis.set_ylabel('Predicted (m$^3$ m$^{-3}$)', fontsize=7.2, labelpad=2.0)
            axis.legend(loc='upper left', bbox_to_anchor=(0.05, 1.0), fontsize=6.4, handlelength=1.5, handletextpad=0.45, labelspacing=0.32, borderaxespad=0.0, borderpad=0.0)
        else:
            axis.set_yticklabels([])
        axis.text(0.955, 0.045, f"R {pooled['R']:.2f}\nRMSE {pooled['RMSE']:.3f}", transform=axis.transAxes, va='bottom', ha='right', fontsize=6.4, color=st.INK, linespacing=1.38, path_effects=HALO)
        lower = figure.add_subplot(grid[1, index], sharex=axis)
        if index == 0:
            first_bottom = lower
        observed = smooth(hist.sum(1) / (total * width))
        predicted = smooth(hist.sum(0) / (total * width))
        lower.fill_between(centres, observed, color=OBS_INK, alpha=0.16, lw=0)
        lower.fill_between(centres, predicted, color=spec['line'], alpha=0.2, lw=0)
        lower.plot(centres, observed, color=OBS_INK, lw=LINE, label='observed')
        lower.plot(centres, predicted, color=spec['line'], lw=LINE, label='predicted')
        lower.set_xlim(lo, hi)
        ceiling = max(observed.max(), predicted.max()) if args.free_y else density_max
        lower.set_ylim(0, ceiling * 1.12)
        lower.set_xticks([0, 0.2, 0.4, 0.6])
        lower.yaxis.set_major_locator(MaxNLocator(4, prune='upper'))
        lower.tick_params(length=2.6, width=LINE, pad=1.8)
        if index == 0:
            lower.set_ylabel('Density', fontsize=7.2, labelpad=2.0)
            lower.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=6.4, handlelength=1.5, handletextpad=0.45, labelspacing=0.32, borderaxespad=0.0, borderpad=0.0)
        elif not args.free_y:
            lower.set_yticklabels([])
        for side in ('top', 'right'):
            lower.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            lower.spines[side].set_linewidth(LINE)
            lower.spines[side].set_color(FRAME)
    print(f'  {args.protocol}: suppression below {VMIN_HEX} station-days/hexagon')
    for label, result in suppression:
        print(
            f"    {label:9s} {result['removed']:.0f}/{result['occupied']:.0f} "
            f"occupied hexagons ({result['hex_pct']:.3f}%); "
            f"{result['station_day_pct']:.6f}% of station-days"
        )
    print(
        f"    range     {min(r['hex_pct'] for _, r in suppression):.3f}-"
        f"{max(r['hex_pct'] for _, r in suppression):.3f}% of occupied hexagons; "
        f"{min(r['station_day_pct'] for _, r in suppression):.6f}-"
        f"{max(r['station_day_pct'] for _, r in suppression):.6f}% of station-days"
    )
    vmax = max((float(np.asarray(h.get_array()).max()) for h in hexes))
    norm = LogNorm(vmin=VMIN_HEX, vmax=vmax)
    for hexed in hexes:
        hexed.set_array(np.ma.masked_less(np.asarray(hexed.get_array(), dtype=float), VMIN_HEX))
        hexed.set_norm(norm)
    cax = figure.add_subplot(grid[0, len(WIDTH_RATIOS) - 1])
    bar = figure.colorbar(hexes[-1], cax=cax)
    bar.set_label('Station-days per hexagon', fontsize=6.8, labelpad=3.0)
    bar.ax.tick_params(labelsize=6.2, length=2.4, width=LINE, pad=1.5)
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(LINE)
    bar.outline.set_edgecolor(FRAME)
    figure.supxlabel('Observed (m$^3$ m$^{-3}$)', fontsize=7.2, y=(MARGIN_BOTTOM - 0.3) / figure_h)
    if spec['banner']:
        first_top.text(0.0, 1.12, spec['banner'], transform=first_top.transAxes, va='bottom', ha='left', fontsize=7.4, color=spec['line'], fontweight='bold')
    st.panel_label(first_top, 'a', x=-0.34, y=1.01)
    st.panel_label(first_bottom, 'b', x=-0.34, y=1.01)
    figure.canvas.draw()
    for name, target in (('a', first_top), ('b', first_bottom)):
        box = target.get_position()
        drawn_w, drawn_h = (box.width * st.WIDTH_DOUBLE, box.height * figure_h)
        if abs(drawn_w - drawn_h) > 0.005:
            raise RuntimeError(f'panel {name} is not square: {drawn_w:.4f} x {drawn_h:.4f} in (expected {panel_w:.4f}); check the margins and gridspec ratios')
    st.save(figure, spec['out'])
    return 0
if __name__ == '__main__':
    sys.exit(main())
