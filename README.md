# FutureTST Hourly Streamflow Forecasting Demo

Multi-step hourly streamflow forecasting on CAMELS-H data with FutureTST, an
encoder-decoder Transformer that conditions on known future meteorological
inputs. The pipeline covers preprocessing, training, and evaluation.

## Layout

```
futuretstdemo/
├── run_forecast.sh                  # runs the full pipeline
├── data/
│   └── camelsh_demo.parquet         # demo data: 5 basins, hourly, 1990-2022
├── data_processing/
│   ├── preprocess_camelsh_forecast.py   # parquet -> prepped.npz
│   └── postprocess_forecast.py          # denormalization + NSE/KGE/RMSE metrics + plots
└── futuretst/
    ├── requirements.txt
    └── src/                         # FutureTST model and training code
```

## Setup

```bash
pip install -r futuretst/requirements.txt
```

A GPU is expected (`--device cuda` by default).

## Run

```bash
bash run_forecast.sh
```

Defaults: 168-hour input window, 18-hour forecast horizon, 200 training epochs
with early stopping (patience 20).

Common variations:

```bash
bash run_forecast.sh --pred_len 24                     # 24-hour horizon
bash run_forecast.sh --windows 336 --pred_len 48       # longer window
bash run_forecast.sh --parquet /path/to/your.parquet   # different data
bash run_forecast.sh --basins "03550000 03453500"      # subset of basins
bash run_forecast.sh --epochs 5                        # quick smoke test
CUDA_VISIBLE_DEVICES=1 bash run_forecast.sh            # pick a GPU
```

## Data format

The input parquet has one row per (basin, hour) with columns:

- `basin_id`, `Time`
- 11 dynamic meteorological features: `CAPE, CRainf_frac, LWdown, PotEvap, PSurf, Qair, Rainf, SWdown, Tair, Wind_E, Wind_N`
- 24 static basin attributes: `p_mean, pet_mean, aridity, ...` (see `STATIC_VARS` in `preprocess_camelsh_forecast.py`)
- target: `Q_camelsh_obs_norm` (area-normalized streamflow)
- `latitude, longitude` (used for the basin distance matrix)

Preprocessing adds 24/72/168-hour cumulative rainfall, mean air temperature,
and cumulative potential evaporation, for 44 input features in total.

## Basin selection

The full study trains a single FutureTST model jointly on 618 basins:
the 130 basins of the Tennessee Valley region plus 488 auxiliary basins
drawn from the full CAMELS-H archive (~9,000 gauges). Auxiliary basins were
required to have at least 97% hourly streamflow completeness and were then
ranked by hydroclimatic similarity to the Tennessee Valley basins using
standard catchment attributes (aridity index, mean precipitation, snow
fraction, elevation, slope, drainage area). This supplies the model with
additional training basins whose rainfall-runoff behavior is transferable
to the Tennessee Valley region while keeping noisy hourly records out of
training.

The bundled demo dataset (`data/camelsh_demo.parquet`) is a 5-basin subset
of the Tennessee Valley basins, chosen for their high streamflow
observation coverage, so the repository can be cloned and run end-to-end
without downloading the full dataset.

Date splits (editable at the top of `preprocess_camelsh_forecast.py`):

- Train: 1997-01-01 to 2018-12-31
- Val:   1995-01-01 to 1996-12-31
- Test:  2019-01-01 to 2022-12-31

## Outputs

- `futuretst/output/pred/tst.npy` — normalized sliding-window predictions (stride 24 by default)
- `futuretst/output/denorm/` — denormalized prediction vs observation CSVs and per-basin metrics (NSE, KGE, RMSE, ...)
- `futuretst/output/figure/` — prediction vs observation plots per basin
- `futuretst/results/` — model checkpoints
