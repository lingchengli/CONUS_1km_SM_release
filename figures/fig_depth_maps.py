#!/usr/bin/env python3
"""Plot monthly mean soil moisture for the five released depth layers."""
from __future__ import annotations
import argparse
import calendar
import datetime as dt
import os
from pathlib import Path
import cartopy.crs as ccrs
import plot_style as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
HALO = [patheffects.withStroke(linewidth=1.6, foreground='white')]
DATA_ROOT = Path(os.environ.get('CONUSSM_PRODUCT_DATA', Path(__file__).resolve().parents[1] / 'product_data'))
PRODUCT = DATA_ROOT / 'product'
MASK = DATA_ROOT / 'conus_admin_mask_1k.nc'
PMASK = DATA_ROOT / 'predictable_mask_2001_2025.nc'
CLIM_FIELDS = DATA_ROOT / 'depth_climatology.npz'
LAYERS = [('L1', '0–5 cm'), ('L2', '5–15 cm'), ('L3', '15–30 cm'), ('L4', '30–60 cm'), ('L5', '60–100 cm')]
DEFAULT_MONTH = '201207'
EXPECTED_LAT_ENDPOINTS = (49.49583333333334, 24.004166666666663)
EXPECTED_LON_ENDPOINTS = (-124.99583333333334, -66.50416666666666)
STRIDE = 3
VMIN, VMAX = (0.0, 0.5)
CMAP = 'YlGnBu'

def smooth_density(density, sigma_bins: float=5.0):
    """A light Gaussian taper, not a kernel density estimate."""
    half = int(4 * sigma_bins)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_bins) ** 2)
    return np.convolve(density, kernel / kernel.sum(), mode='same')

def predictable_cells(year: int) -> int:
    """How many valid cells a month of `year` must carry."""
    with nc.Dataset(PMASK) as dataset:
        years = np.asarray(dataset.variables['year'][:])
        hit = np.where(years == year)[0]
        if hit.size != 1:
            raise RuntimeError(f'{PMASK}: no single {year} slice; has {years.min()}-{years.max()}')
        return int(np.asarray(dataset.variables['predictable'][int(hit[0])]).sum())

def climatology_spec() -> dict:
    """The 2001-2025 mean field, on the same axes as a single month."""
    return {'yyyymm': None, 'year': None, 'month': None, 'days': None, 'valid': None, 'label': '2001–2025 mean', 'out': 'depth_maps_climatology'}

def month_spec(yyyymm: str) -> dict:
    """Everything that varies with the displayed month, derived not restated."""
    if len(yyyymm) != 6 or not yyyymm.isdigit():
        raise SystemExit(f'--month wants YYYYMM, got {yyyymm!r}')
    year, month = (int(yyyymm[:4]), int(yyyymm[4:]))
    if not 1 <= month <= 12:
        raise SystemExit(f'--month has no month {month}')
    return {'yyyymm': yyyymm, 'year': year, 'month': month, 'label': dt.date(year, month, 1).strftime('%B %Y'), 'days': calendar.monthrange(year, month)[1], 'valid': predictable_cells(year), 'out': 'depth_maps' if yyyymm == DEFAULT_MONTH else f'depth_maps_{yyyymm}'}

def read_layer(layer: str, spec: dict):
    """Monthly mean of one layer, with the grid checked against the record."""
    path = PRODUCT / f"sm_{layer}_{spec['yyyymm']}.nc"
    with nc.Dataset(path) as dataset:
        stack = np.ma.filled(dataset.variables[f'sm_{layer}'][:], np.nan)
        lat = np.asarray(dataset.variables['lat'][:])
        lon = np.asarray(dataset.variables['lon'][:])
        conventions = dataset.Conventions
        time = dataset.variables['time']
        stamps = nc.num2date(time[:], units=time.units, calendar=getattr(time, 'calendar', 'standard'))
    if len(stamps) != spec['days']:
        raise RuntimeError(f"{path}: {len(stamps)} steps, not {spec['days']}")
    if not all((s.year == spec['year'] and s.month == spec['month'] for s in stamps)):
        raise RuntimeError(f"{path}: time axis is not entirely {spec['label']}")
    if conventions != 'CF-1.9':
        raise RuntimeError(f'{path}: unexpected convention {conventions}')
    if not np.allclose([lat[0], lat[-1]], EXPECTED_LAT_ENDPOINTS, rtol=0, atol=1e-10):
        raise RuntimeError(f'{path}: unexpected latitude endpoints')
    if not np.allclose([lon[0], lon[-1]], EXPECTED_LON_ENDPOINTS, rtol=0, atol=1e-10):
        raise RuntimeError(f'{path}: unexpected longitude endpoints')
    field = np.nanmean(stack, axis=0)
    if int(np.isfinite(field).sum()) != spec['valid']:
        raise RuntimeError(
            f"{path}: {int(np.isfinite(field).sum())} valid cells != "
            f"{spec['valid']} predictable in {spec['year']}"
        )
    return (field, lat, lon)

def conus_mask(lat, lon):
    """The administrative CONUS mask, aligned BY COORDINATE."""
    with xr.open_dataset(MASK) as m:
        mk = m['conus'].reindex(lat=lat, lon=lon, method='nearest', tolerance=0.0001)
    if bool(mk.isnull().any()):
        raise RuntimeError('CONUS mask does not cover the product grid')
    inside = mk.values.astype(bool)

    def at(la, lo):
        return bool(inside[int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())])
    for la, lo, want, where in ((47.4, -122.0, True, 'Bellevue WA'), (44.8, -68.8, True, 'Bangor ME'), (26.0, -105.0, False, 'Chihuahua MX'), (49.9, -110.0, False, 'Alberta CA')):
        if at(la, lo) is not want:
            location = "outside" if want else "inside"
            raise RuntimeError(
                f"CONUS mask orientation wrong: {where} ({la}, {lo}) is {location}"
            )
    return inside

def main(spec: dict):
    st.apply()
    proj = ccrs.AlbersEqualArea(central_longitude=-96, standard_parallels=(29.5, 45.5))
    fields, lat, lon = ([], None, None)
    if spec['yyyymm'] is None:
        if not CLIM_FIELDS.exists():
            raise SystemExit(f'missing {CLIM_FIELDS}')
        store = np.load(CLIM_FIELDS)
        lat, lon = (store['lat'], store['lon'])
        fields = [store[layer] for layer, _ in LAYERS]
    else:
        for layer, _ in LAYERS:
            field, la, lo = read_layer(layer, spec)
            if lat is None:
                lat, lon = (la, lo)
            elif not (np.array_equal(la, lat) and np.array_equal(lo, lon)):
                raise RuntimeError(f'{layer}: grid differs from L1')
            fields.append(field)
    inside = conus_mask(lat, lon)
    fields = [np.where(inside, f, np.nan) for f in fields]
    means = [float(np.nanmean(f)) for f in fields]
    aspect = st.CONUS_ASPECT
    LEFT, RIGHT, WSP, HSP = (0.055, 0.91, 0.1, 0.22)
    TOP_IN, BOT_IN = (0.2, 0.1)
    cell_w = (RIGHT - LEFT) * st.WIDTH_DOUBLE / (3 + 2 * WSP)
    cell_h = cell_w / aspect
    height = cell_h * (2 + HSP) + TOP_IN + BOT_IN
    fig = plt.figure(figsize=(st.WIDTH_DOUBLE, height))
    grid = fig.add_gridspec(2, 3, left=LEFT, right=RIGHT, top=1 - TOP_IN / height, bottom=BOT_IN / height, wspace=WSP, hspace=HSP)
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    for index, ((layer, depth), field) in enumerate(zip(LAYERS, fields)):
        row, col = divmod(index, 3)
        axis = fig.add_subplot(grid[row, col], projection=proj)
        axis.pcolormesh(lon[::STRIDE], lat[::STRIDE], field[::STRIDE, ::STRIDE], shading='auto', cmap=CMAP, norm=norm, transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
        st.conus_frame(axis, left_labels=col == 0, bottom_labels=row == 1)
        axis.set_title(depth, fontsize=8.0, pad=4)
        st.panel_label(axis, chr(97 + index), x=-0.01, y=1.02)
        axis.text(0.015, 0.04, f'mean = {means[index]:.2f} m$^3$ m$^{{-3}}$', transform=axis.transAxes, ha='left', va='bottom', fontsize=7.2, color=st.INK, path_effects=HALO, zorder=4)
    fax = fig.add_subplot(grid[1, 2])
    edges = np.linspace(VMIN, VMAX, 501)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for (layer, depth), field, colour in zip(LAYERS, fields, st.DEPTH_COLORS):
        density, _ = np.histogram(field[np.isfinite(field)], bins=edges, density=True)
        fax.plot(centres, smooth_density(density), color=colour, lw=0.9, label=depth.replace(' cm', ''), solid_capstyle='round')
    fax.set_xlim(VMIN, VMAX)
    fax.set_ylim(bottom=0)
    fax.set_xlabel('Soil moisture (m$^3$ m$^{-3}$)', fontsize=7.2, labelpad=2)
    fax.set_ylabel('Density', fontsize=7.2, labelpad=3)
    fax.yaxis.set_label_position('right')
    fax.yaxis.tick_right()
    fax.tick_params(labelsize=6.8, length=2.4, width=st.LINE, color=st.INK)
    for side in ('top', 'right', 'bottom', 'left'):
        fax.spines[side].set_visible(True)
        fax.spines[side].set_linewidth(st.LINE)
        fax.spines[side].set_edgecolor(st.INK)
    fax.set_title(spec['label'], fontsize=8.0, pad=4)
    st.panel_label(fax, 'f', x=-0.01, y=1.02)
    fax.legend(fontsize=5.8, frameon=False, handlelength=0.9, labelspacing=0.22, borderpad=0.1, loc='upper right', handletextpad=0.4, title='Depth (cm)', title_fontsize=6.0)
    ccell = grid[0, 2].get_position(fig)
    cax = fig.add_axes([ccell.x1 + 0.014, ccell.y0, 0.012, ccell.height])
    clipped = max((float(np.nanmax(f)) for f in fields)) > VMAX
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation='vertical', extend='max' if clipped else 'neither')
    bar.set_label('Soil moisture (m$^3$ m$^{-3}$)', fontsize=7.6, labelpad=4)
    bar.ax.tick_params(labelsize=6.8, length=2.4, width=st.LINE)
    bar.outline.set_linewidth(st.LINE)
    bar.outline.set_edgecolor(st.INK)
    st.save(fig, spec['out'])
    print('  CONUS cells per layer:', int(inside.sum()))
    print('  domain means:', ', '.join((f'{v:.3f}' for v in means)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month', default=DEFAULT_MONTH, metavar='YYYYMM', help='month to render (default %(default)s)')
    parser.add_argument('--climatology', action='store_true', help='draw the 2001-2025 mean field instead of a month')
    args = parser.parse_args()
    main(climatology_spec() if args.climatology else month_spec(args.month))
