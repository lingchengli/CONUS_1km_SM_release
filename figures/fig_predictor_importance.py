#!/usr/bin/env python3
"""Plot relative SHAP importance by predictor family and predictor."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as st
DATA_DIR = Path(os.environ.get('CONUSSM_FIGURE_DATA', Path(__file__).resolve().parents[1] / 'figure_data'))
LAYERS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
TITLES = {'0-5cm': '0-5 cm', '5-15cm': '5-15 cm', '15-30cm': '15-30 cm', '30-60cm': '30-60 cm', '60-100cm': '60-100 cm'}
FAMILIES = ['Hydrothermal conditions', 'Meteorology', 'Vegetation', 'Soil property', 'Topography', 'Season']
COLORS = dict(zip(FAMILIES, ['#0072B2', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#E69F00']))
LINE = st.LINE

def load():
    path = DATA_DIR / 'shap_summary.json'
    if not path.exists():
        raise SystemExit(f'missing {path}; set CONUSSM_FIGURE_DATA to the derived-data directory')
    return json.loads(path.read_text())

def family_of(feature: str) -> str:
    if feature in {'ET', 'SWE', 'WTD'} or feature.startswith(('SM_l', 'ST_l')):
        return 'Hydrothermal conditions'
    if feature in {'DAYL', 'PRCP', 'SRAD', 'TMAX', 'TMIN', 'VP'} or feature.startswith(('PRCP_', 'SRAD_')):
        return 'Meteorology'
    if feature in {'LAI', 'RD', 'PFT'}:
        return 'Vegetation'
    if feature.startswith(('ORGANIC_l', 'PCT_CLAY_l', 'PCT_SAND_l')):
        return 'Soil property'
    if feature in {'ELEVATION', 'SLOPE', 'ASPECT', 'STDEV_ELEV'}:
        return 'Topography'
    if feature in {'doy_sin', 'doy_cos'}:
        return 'Season'
    raise ValueError(f'unrecognized predictor: {feature}')

def _bars(axis, labels, values, colors, value_fmt='{:.1f}'):
    """Horizontal bars, largest at the TOP, with the value printed at the end."""
    y = np.arange(len(labels))[::-1]
    axis.barh(y, values, height=0.72, color=colors, linewidth=0)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.set_ylim(-0.7, len(labels) - 0.3)
    axis.set_xlim(0, max(values) * 1.18)
    for yi, v in zip(y, values):
        axis.text(v + max(values) * 0.02, yi, value_fmt.format(v), va='center', ha='left', fontsize=6.2, color=st.MUTED)
    for side in ('left', 'bottom'):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_linewidth(LINE)
        axis.spines[side].set_color(st.INK)
    axis.tick_params(axis='x', length=2.2, width=LINE)
    axis.tick_params(axis='y', length=0)

def figure_groups(report) -> None:
    """Main text: six groups per layer, ranked, one panel per layer."""
    st.apply()
    fig, axes = plt.subplots(2, 3, figsize=(st.WIDTH_DOUBLE, 3.5))
    for axis, layer in zip(axes.ravel(), LAYERS):
        pct = report[layer]['pct_by_family']
        order = sorted(FAMILIES, key=lambda g: -pct[g])
        _bars(axis, order, [pct[g] for g in order], [COLORS[g] for g in order])
        axis.set_title(TITLES[layer], fontsize=8.0, pad=4)
        axis.set_xlabel('Relative importance (%)', fontsize=6.8)
    spare = axes.ravel()[len(LAYERS)]
    spare.axis('off')
    sizes = report[LAYERS[0]]['family_sizes']
    spare.legend(handles=[Patch(facecolor=COLORS[g], label=f'{g} (n={sizes[g]})') for g in FAMILIES], loc='center', frameon=False, fontsize=7.0, title='Predictor group', title_fontsize=7.5)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    st.save(fig, 'predictor_group_importance')

def figure_features(report) -> None:
    """Supplementary: all 53 predictors per layer, coloured by group."""
    st.apply()
    fig, axes = plt.subplots(1, 5, figsize=(st.WIDTH_DOUBLE, 8.6))
    for axis, layer in zip(axes, LAYERS):
        pct = report[layer]['pct_by_feature']
        names = list(pct)
        _bars(axis, names, [pct[f] for f in names], [COLORS[family_of(f)] for f in names], value_fmt='{:.2f}')
        axis.set_title(TITLES[layer], fontsize=7.5, pad=4)
        axis.tick_params(axis='y', labelsize=4.6)
        axis.tick_params(axis='x', labelsize=5.6)
        axis.set_xlabel('Relative\nimportance (%)', fontsize=6.0)
    fig.legend(handles=[Patch(facecolor=COLORS[g], label=g) for g in FAMILIES], loc='lower center', ncol=6, frameon=False, fontsize=7.0, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(w_pad=1.0, rect=(0, 0.022, 1, 1))
    st.save(fig, 'individual_predictor_importance')

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--groups', action='store_true', help='main-text figure only')
    ap.add_argument('--features', action='store_true', help='supplementary figure only')
    args = ap.parse_args()
    report = load()
    both = not (args.groups or args.features)
    if args.groups or both:
        figure_groups(report)
    if args.features or both:
        figure_features(report)
    return 0
if __name__ == '__main__':
    sys.exit(main())
