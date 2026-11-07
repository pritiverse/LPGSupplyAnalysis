# LPG Supply Analysis

Tools for exploring LPG supply-chain shocks and identifying unusual country-level supply profiles. The repository combines Jupyter Notebook analysis with Python data-processing scripts and interactive Streamlit dashboards. The dashboard describes coverage of 172 countries from 1990–2024; actual results depend on the data and checkpoints available in your environment.

## What’s included

- **Supply-chain shock dashboard** (`lpg_dashboard/`) for global, country, event, model, and detection views.
- **Structural anomaly analysis** using Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM (OCSVM), with a composite score.
- **Structural anomaly dashboard** (`streamlit_structural_app.py`) with interactive charts and model controls.
- **Analysis notebook** (`code_googlecollab.ipynb`) and supporting Python scripts for analysis and checkpoint generation.

## Requirements

Use Python 3. A typical environment for the applications and analysis includes:

- `streamlit`
- `pandas` and `numpy`
- `scikit-learn`
- `plotly`
- `pyarrow` (for Parquet files)

Some scripts or notebook cells may require additional packages. Install those in your environment as needed.

## Run the dashboards

From the repository root, install the packages above and start the desired dashboard:

```bash
streamlit run lpg_dashboard/app.py
```

For the structural anomaly dashboard:

```bash
streamlit run streamlit_structural_app.py
```

The dashboards expect prepared data checkpoints. In particular, the structural anomaly dashboard loads Parquet data from `pipeline_checkpoints/country_pca_features/`; the supply-chain dashboard also expects its pipeline checkpoints. These generated data and model artifacts are not included in the repository (the data directories are git-ignored). Generate or provide the required checkpoints before launching the dashboards.

## Structural anomaly workflow

The standalone structural analysis script reads `pipeline_checkpoints/country_pca_features/`, fits the three detectors, and writes results under `pipeline_checkpoints/structural_anomaly_detection/` and `outputs/anomalies/`:

```bash
python newapp.py
```

The script also saves a visualization as `structural_anomalies.png`. For the interactive dashboard, run `streamlit_structural_app.py` as shown above.

## Data and interpretation

Model outputs are exploratory indicators of unusual profiles, not definitive findings about a country's supply security. Results depend on input data, feature selection, scaling, and model parameters. Review the underlying data and model assumptions before drawing conclusions.

## Repository contents

- `lpg_dashboard/` — supply-chain shock dashboard
- `pipeline_checkpoints/` — generated intermediate datasets (not committed)
- `raw_data/`, `outputs/`, `models/` — data, results, and model artifacts (not committed)
- `code_googlecollab.ipynb` — notebook workflow
- `streamlit_structural_app.py` — interactive structural anomaly dashboard
- `newapp.py` — structural anomaly analysis and output generation
- `Composite_Structural_Score.py` — composite score generation from model checkpoints
