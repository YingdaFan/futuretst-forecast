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

## Full datasets

The full datasets are available for download
[here (OneDrive)](https://rutgersconnect-my.sharepoint.com/:u:/g/personal/yf474_scarletmail_rutgers_edu/IQAWuS4GdUGjRIDdYdF4vsj7AVsXSWJKG7WeU-lREIeBG0w):

- `camelsh_tennessee.parquet` — the 130 Tennessee Valley basins (~1.3 GB)
- `camelsh_global.parquet` — all 618 basins (130 TVA + 488 auxiliary, ~6.5 GB)

The same pipeline reproduces the full experiments by pointing `--parquet` at
the downloaded file:

```bash
bash run_forecast.sh --parquet /path/to/camelsh_tennessee.parquet   # 130 TVA basins
bash run_forecast.sh --parquet /path/to/camelsh_global.parquet      # all 618 basins
```

Note that training on the full datasets is substantially heavier than the
demo: preprocessing the 618-basin dataset needs tens of GB of RAM, and one
training batch holds one time window for every basin.

Date splits (editable at the top of `preprocess_camelsh_forecast.py`):

- Train: 1997-01-01 to 2018-12-31
- Val:   1995-01-01 to 1996-12-31
- Test:  2019-01-01 to 2022-12-31

## Outputs

- `futuretst/output/pred/tst.npy` — normalized sliding-window predictions (stride 24 by default)
- `futuretst/output/denorm/` — denormalized prediction vs observation CSVs and per-basin metrics (NSE, KGE, RMSE, ...)
- `futuretst/output/figure/` — prediction vs observation plots per basin
- `futuretst/results/` — model checkpoints

## FAQ

**What model is this?**
FutureTST, an encoder-decoder Transformer for time series forecasting. Its
distinguishing feature is that the decoder attends to the exogenous inputs
over the forecast horizon as well as the history, so known future
meteorological forcings inform the streamflow prediction. There is no
diffusion component.

**Does the code download CAMELS-H data automatically?**
No. Everything runs offline. The repo ships with the 5-basin demo parquet,
and `run_forecast.sh` takes it through preprocessing, training, evaluation,
and plotting. For the full experiments, download a dataset from the OneDrive
link above and pass it with `--parquet`.

**How many basins were used in the actual study?**
Two configurations: a single model trained jointly on the 130 Tennessee
Valley basins, and a single model trained jointly on 618 basins (the 130 TVA
basins plus 488 auxiliary basins; see "Basin selection"). Both datasets are
on OneDrive and both runs use the same script, differing only in the
`--parquet` argument. The 5 demo basins are TVA basins with high observation
coverage.
