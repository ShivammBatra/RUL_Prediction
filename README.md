# Capstone: Turbofan Remaining Useful Life

This repository contains the capstone source code, experiment notebooks, dashboard, and selected result files for Remaining Useful Life (RUL) prediction and reliability analysis on the NASA C-MAPSS turbofan engine data.

## Repository layout

- `dashboard_app.py`, `live_inference.py`: Streamlit dashboard and saved-model inference helpers.
- `requirements-dashboard.txt`: dashboard Python dependencies.
- `notebooks/analysis/`: preprocessing, baseline, reliability-gate, sensor-consistency, and evaluation notebooks.
- `notebooks/capped/`: capped-RUL experiments.
- `notebooks/uncapped/`: uncapped-RUL experiments.
- `results/`: saved experiment metrics, prediction tables, and plots. Some report documents are kept local and excluded from GitHub.
- `fault_detection_results.csv`, `reliability_gate_results.csv`, `reliability_gate_clean_results.csv`: dashboard fault-detection and reliability-gate inputs.

## Run the dashboard

Use Python 3.10 or another version supported by the pinned TensorFlow range in `requirements-dashboard.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dashboard.txt
python -m streamlit run dashboard_app.py
```

The dashboard's data and live-inference pages require local files that are intentionally excluded from GitHub: `Dataset/`, `bilstm_reliability_gate.keras`, and `consistency_models.pkl`. Without those files, the dashboard can show missing-data notices and the live model demo will not run. Place the corresponding project files in the paths expected by `dashboard_app.py` to use those features.

## Notebooks

The notebooks contain experiment-specific cells and some existing machine-local data paths. They may need dataset paths adjusted for another computer. The source datasets and generated intermediate datasets are not committed.

## Scope

This repository is a capstone research/demo project. Its experimental results are limited to the datasets and synthetic corruption/fault scenarios documented in the notebooks; they should not be treated as operational certification.