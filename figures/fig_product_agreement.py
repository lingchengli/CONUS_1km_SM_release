#!/usr/bin/env python3
"""Plot grid-to-grid anomaly agreement and mean signed differences."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import cartopy.io.shapereader as shpreader
from rasterio import features
from affine import Affine
from shapely.ops import unary_union
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as st
from fig_surface_comparison import RAMP
METRICS = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
EXTENT = (-125.0, -66.5, 24.0, 49.5)
COLUMNS = [('ours_vs_swsm_sfc', 'SWSM', None), ('ours_vs_era5_sfc', 'ERA5-Land', None), ('ours_vs_gleam_sfc', 'GLEAM', None), ('ours_vs_cci_sfc', 'ESA CCI', None)]
R_VMIN, R_VMAX = (0.3, 1.0)
D_ABS = 0.1

def conus_mask(lat, lon):
    """CONUS mask on the comparison grid."""
    reader = shpreader.Reader(shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries'))
    geom = unary_union([r.geometry for r in reader.records() if r.attributes.get('ADM0_A3') == 'USA'])
    dlon = float(lon[1] - lon[0])
    dlat = float(abs(lat[1] - lat[0]))
    north_up = lat[0] > lat[-1]
    top = float(lat[0] if north_up else lat[-1]) + dlat / 2
    tf = Affine.translation(float(lon[0]) - dlon / 2, top) * Affine.scale(dlon, -dlat)
    burned = features.rasterize([(geom, 1)], out_shape=(lat.size, lon.size), transform=tf, fill=0, dtype='uint8')
    return (burned if north_up else np.flipud(burned)).astype(bool)

def base_map(axis, lon, lat, field=None, left=True, bottom=True):
    st.conus_frame(axis, left_labels=left, bottom_labels=bottom)

def main() -> int:
    data = {}
    for key, _, _ in COLUMNS:
        if key is not None:
            data[key] = xr.open_dataset(METRICS / f'metrics_conus_{key}.nc')
    ref = next(iter(data.values()))
    for key, d in data.items():
        if not (np.array_equal(d['lat'].values, ref['lat'].values) and np.array_equal(d['lon'].values, ref['lon'].values)):
            raise RuntimeError(f'{key}: coordinates differ from the reference grid; these rasters are combined positionally')
    keep = conus_mask(ref['lat'].values, ref['lon'].values)
    for d in data.values():
        keep &= np.isfinite(d['Ranom'].values)
        keep &= np.isfinite(d['mean_signed_difference'].values)
    st.apply()
    cmap_r = LinearSegmentedColormap.from_list('ramp', RAMP)
    cmap_d = LinearSegmentedColormap.from_list('div', ['#8C510A', '#D8B365', '#F5F5F5', '#5AB4AC', '#01665E'])
    nr = Normalize(R_VMIN, R_VMAX)
    nd = TwoSlopeNorm(vmin=-D_ABS, vcenter=0.0, vmax=D_ABS)
    proj = ccrs.AlbersEqualArea(central_longitude=-96, standard_parallels=(29.5, 45.5))
    LEFT, RIGHT, WSP = (0.045, 0.915, 0.05)
    cell_w = (RIGHT - LEFT) * st.WIDTH_DOUBLE / (4 + 3 * WSP)
    cell_h = cell_w / st.CONUS_ASPECT
    TOP_IN, GAP_IN, BOT_IN = (0.34, 0.36, 0.26)
    height = TOP_IN + 2 * cell_h + GAP_IN + BOT_IN
    fig = plt.figure(figsize=(st.WIDTH_DOUBLE, height))
    row_a = fig.add_gridspec(1, 4, left=LEFT, right=RIGHT, top=1 - TOP_IN / height, bottom=1 - (TOP_IN + cell_h) / height, wspace=WSP)
    row_b = fig.add_gridspec(1, 4, left=LEFT, right=RIGHT, top=(BOT_IN + cell_h) / height, bottom=BOT_IN / height, wspace=WSP)
    handles = {}
    for c, (key, label, blank) in enumerate(COLUMNS):
        for row, grid, var, cmap, norm in [(0, row_a, 'Ranom', cmap_r, nr), (1, row_b, 'mean_signed_difference', cmap_d, nd)]:
            axis = fig.add_subplot(grid[0, c], projection=proj)
            if key is None:
                axis.set_extent(EXTENT, crs=ccrs.PlateCarree())
                axis.text(0.5, 0.5, blank, transform=axis.transAxes, ha='center', va='center', fontsize=5.4, color='#A0A09C', linespacing=1.5)
                axis.spines['geo'].set_edgecolor('#E2E2DE')
                axis.spines['geo'].set_linewidth(st.LINE)
            else:
                d = data[key]
                base_map(axis, d['lon'].values, d['lat'].values, field=np.where(keep, 1.0, np.nan), left=c == 0, bottom=True)
                field = np.where(keep, d[var].values, np.nan)
                handles[row] = axis.pcolormesh(d['lon'], d['lat'], field, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto', zorder=2, rasterized=True)
                axis.text(0.02, 0.05, f'median {np.nanmedian(field):+.3f}', transform=axis.transAxes, fontsize=7.0, color=st.INK, ha='left', va='bottom', zorder=5, path_effects=[patheffects.withStroke(linewidth=1.6, foreground='white')])
            if row == 0:
                axis.set_title(f'ConusSM1k − {label}', fontsize=7.2, pad=4)
                if c == 0:
                    st.panel_label(axis, 'a', x=-0.03, y=1.02)
            elif c == 0:
                st.panel_label(axis, 'b', x=-0.03, y=1.02)
    for grid, row, label, extend in [(row_a, 0, 'anomaly R', 'min'), (row_b, 1, 'mean difference (m$^3$ m$^{-3}$)', 'both')]:
        cell = grid[0, 3].get_position(fig)
        cax = fig.add_axes([cell.x1 + 0.012, cell.y0, 0.011, cell.height])
        cb = fig.colorbar(handles[row], cax=cax, orientation='vertical', extend=extend)
        cb.set_label(label, fontsize=6.8, labelpad=4)
        cb.ax.tick_params(labelsize=6.2, width=st.LINE, length=2.2)
        cb.outline.set_linewidth(st.LINE)
        cb.outline.set_edgecolor(st.INK)
    st.save(fig, 'product_agreement')
    return 0
if __name__ == '__main__':
    sys.exit(main())
