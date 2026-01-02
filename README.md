# Geometry-Based RF Wireless Power Transfer

Python tooling to coordinate a geometry-aware wireless power transfer experiment over distributed tiles (Raspberry Pi + USRP B210) using a ZMQ control plane and tile-management Ansible playbooks.

## Repository layout
- `experiment-settings.yaml`: central experiment config (tile groups, RF params, server host, client script + args).
- `server/`: control-node tooling for provisioning tiles, updating experiment artifacts, starting/stopping clients, and the ZMQ coordinator.
- `client/`: scripts that run on the tiles plus calibration data for the USRPs (`cal-settings.yml`) and best-phase snapshots (`tx-phases-energy-ball.yml`).
- `processing/`: helper scripts used to prepare experiment inputs (e.g., TX phase generation) and plot results (matplotlib + Plotly/Dash heatmaps).
- `lib/`: shared helpers (energy profiler, YAML utilities); `pictures/`: diagrams/results.

## Prerequisites
- Control machine with Python 3, Git/SSH access to the tiles, and Ansible available.
- `tile-management` repo checked out at `~/tile-management` (or let `server/setup-server.sh` clone/update it).
- Tiles listed in `~/tile-management/inventory/hosts.yaml` and reachable via SSH; hostnames follow `rpi-<id>` so the client IDs resolve correctly.
- UHD/B210 stack on the tiles (validated by `server/setup-clients.py`).

## Setup (control node)
1) Prep TX phase files:
   - Sionna ray-tracing helper (writes `client/tx-weights-sionna.yml` and `client/tx-phases-sionna-<specular_order>SDR.yml`, e.g., `2SDR` was used previously):
```
cd processing
python compute-tx-weights-sionna.py
cd ..
```
   - Energy-ball workflow: post-process the latest `server/record/data/exp-*.yml` to refresh `client/tx-phases-energy-ball.yml` and visualize iteration power:
```
python processing/process-energy-ball.py --plot
```
2) Bootstrap the virtualenv and pull tile-management:
```
cd server
./setup-server.sh         # clones/updates tile-management and installs deps
source bin/activate
```
3) Configure `experiment-settings.yaml` with the server host/IP, target tile group(s), RF params (`frequency`, `gain`, `rate`, `duration`), `client_script_name`/`client_script_args`, and any extra apt packages.
4) Prepare the tiles (apt update/upgrade, install extras, pull repos, check UHD):
```
python server/setup-clients.py --ansible-output
```
   Flags: `--skip-apt`, `--repos-only`, `--install-only`, `--check-uhd-only` to narrow the actions.
5) Push updated experiment code/settings to the tiles:
```
python server/update-experiment.py --ansible-output
```
6) Start or stop the experiment service on the tiles:
```
python server/run-clients.py --start   # or --stop
```

## Recording measurements
- Use the energy-profiler recorder to capture position/power traces (defaults track `FOLDER` in `server/record/record-meas-energy-profiler.py`, currently `energy-ball-MAX`; update `FOLDER` to start a new series):
```
python server/record/record-meas-energy-profiler.py
```
- Autosaves every `SAVE_EVERY` seconds to `data/<FOLDER>/<timestamp>_{positions,values}.npy` (Qualisys positioning + RFEP power). Update `FOLDER` to start a new series; settings are pulled from `experiment-settings.yaml`.

## Running an experiment
- Launch the ZMQ control server on the control node (with the venv active):
```
python server/run_server.py
```
- Tiles run the client defined in `experiment-settings.yaml` (currently `client/run_energy_ball.py`). The default quasi multi-tone client can be started manually on a tile if needed:
```
python client/run_quasi_multi_tone.py --config-file /home/pi/geometry-based-wireless-power-transfer/experiment-settings.yaml
```
  Clients wait for `tx-start`, transmit for the requested duration, then reply with `tx-done`.
- Energy-ball clients accept `--next-phase-alg` (`random` or `energyball`) to override the YAML setting and read calibration from `cal-settings.yml`.

## Experiment workflows
- **Fixed-phase transmit (`run_gbwt_phases.py`)**
  - Uses the phases defined in `settings.yml`.
  - Requires: `sync-server.py`, the XY plotter (running on TTRPI5), the positioner on the Qualisys server, and `server/record/record-meas-energy-profiler.py`.
  - Plotting: `processing/plot_all_folders_heatmap.py` or `processing/plot-values-positions-2d.py`.

- **Energy-ball (simulated annealing) transmit**
  - Phases are iteratively updated via simulated annealing to converge to an optimal point.
  - Uses the same infrastructure as above (sync server, XY plotter, positioner, recorder) plus `server/record/server-energy-ball.py` (or `server-energy-ball-max.py`) to log per-iteration power to `server/record/data/exp-*.yml`.
  - Clients: `client/run_energy_ball.py` (default) or `client/run_energy_ball_max.py`; both pull initial phases from `tx-phases-energy-ball.yml`.
  - Post-process: `processing/process-energy-ball.py --plot` summarizes the YAML, plots iteration power, and rewrites `client/tx-phases-energy-ball.yml` with the best phases.
  - Plotting: `processing/plot_all_folders_heatmap.py` / `plot-values-positions-2d.py` (matplotlib) or `processing/plot_all_folders_heatmap_live.py` (Plotly/Dash).

## Plotting and post-processing
- Per-folder heatmap (set `FOLDER` inside `processing/plot-values-positions-2d.py`, defaults follow the latest measurement series):
```
cd processing
python plot-values-positions-2d.py
```
- Aggregate folders under `data/` into mean-power heatmaps (newest folder by default; add `--plot-all` for all or `--plot-movement` to outline the last 5 visited cells):
```
python processing/plot_all_folders_heatmap.py
```
- Live Plotly/Dash heatmap that refreshes every 2s (targets newest folder by default):
```
python processing/plot_all_folders_heatmap_live.py --host 0.0.0.0 --port 8050 --target 3.181 1.774 0.266
```
- Energy-ball YAML summarizer (prints/plots max power per iteration; input defaults to latest `server/record/data/exp-*.yml`):
```
python processing/process-energy-ball.py [path/to/exp-YYYYMMDDHHMMSS.yml]
```

## Preparing TX phases
- `processing/compute-tx-phases.py` fetches the tile geometry from `techtile-description` and generates `client/tx-phases-friis.yml` (phase-aligned) and `client/tx-phases-benchmark.yml` (all zeros). Run it from the repo root:
```
python processing/compute-tx-phases.py
```
- `processing/process-energy-ball.py` reads `server/record/data/exp-*.yml` and updates `client/tx-phases-energy-ball.yml` with the best measured phases.

## Maintenance utilities
- `server/cleanup-clients.py`, `server/reboot-clients.py`: quick management helpers for the tiles.
- `client/usrp-cal-bf.py`: USRP calibration/beamforming helper; `ref-RF-cable.yml` and `tx-phases-*.yml` hold calibration data.

## License
MIT (see `LICENSE`).
