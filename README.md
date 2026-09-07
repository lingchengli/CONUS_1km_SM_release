# ConusSM1k

Code accompanying the ConusSM1k data descriptor. The repository provides
model-development and visualization workflows supporting the study.

## Overview

`model/` contains LightGBM workflows for spatially blocked hyperparameter
search and deterministic model fitting. `figures/` contains the plotting
workflows and shared visual configuration used for the study figures.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Model workflow

The model scripts accept a prepared CSV or Parquet table. Feature, target,
station, and coordinate columns are specified at run time. For an independent
spatial evaluation, reserve the outer holdout before hyperparameter search.

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
`--help` for the available options.

## Figures

Configure the figure input and output locations before running a plotting
workflow:

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
