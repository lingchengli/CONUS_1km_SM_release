# ConusSM1k manuscript code

Minimal code release accompanying the ConusSM1k data descriptor. It contains
the manuscript figure builders and two generic examples of the model-selection
and fitting procedure.

The repository intentionally excludes source data, preprocessing, station QA,
production prediction, upload utilities, trained models, the exact predictor
list, selected layer configurations, and search logs.

## Contents

`model/hp_search_example.py` demonstrates FLAML CFO search scored by fivefold
spatially blocked cross-validation. The score gives each station equal weight,
and the final configuration is chosen with a paired block-bootstrap one-standard-
error rule followed by realized model size.

`model/train_example.py` demonstrates a deterministic LightGBM fit using a
configuration selected previously. Both scripts accept a prepared CSV or
Parquet modeling table and user-supplied column names; neither contains
ConusSM1k-specific feature names or fitted parameters.

`figures/` contains the final plotting logic with descriptive filenames:

- `fig_station_support.py`
- `fig_predictor_importance.py`
- `fig_depth_maps.py`
- `fig_native_scale.py`
- `fig_performance.py`
- `fig_surface_comparison.py`
- `fig_product_agreement.py`

The conceptual workflow graphic has no associated analysis code and is not
included.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model examples

The search example expects one row per observation and columns identifying the
station and its coordinates. Feature and target columns are supplied at run
time:

```bash
python model/hp_search_example.py prepared_table.csv \
  --features x1 x2 x3 \
  --target soil_moisture \
  --output output/selected_parameters.json
```

The resulting configuration can be used by the deterministic fit example:

```bash
python model/train_example.py prepared_table.csv \
  output/selected_parameters.json output/model.txt \
  --features x1 x2 x3 \
  --target soil_moisture
```

Use `--categorical column_name ...` when applicable. Run either script with
`--help` for all options.

## Figures

The plotting scripts read already-derived figure inputs. These inputs are not
distributed in this code-only repository. Set their directory and the output
directory before running a builder:

```bash
export CONUSSM_FIGURE_DATA=/path/to/derived_figure_inputs
export CONUSSM_FIGURE_OUTPUT=/path/to/output
python figures/fig_surface_comparison.py
```

The depth-map builder reads the released NetCDF product rather than a figure
cache. Set `CONUSSM_PRODUCT_DATA` to a directory containing:

```text
product/sm_L1_YYYYMM.nc ... product/sm_L5_YYYYMM.nc
conus_admin_mask_1k.nc
predictable_mask_2001_2025.nc
```

Then run:

```bash
export CONUSSM_PRODUCT_DATA=/path/to/product_inputs
python figures/fig_depth_maps.py --month 201207
```

All rendered images are written under `CONUSSM_FIGURE_OUTPUT`, or `output/` by
default.

## License

BSD 3-Clause. See `LICENSE`.
