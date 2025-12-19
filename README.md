# AIEG Forecasting — Electric Load & Solar Production Forecasting

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![macOS](https://img.shields.io/badge/macOS-compatible-000000?logo=apple&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-compatible-FCC624?logo=linux&logoColor=black)
![Windows](https://img.shields.io/badge/Windows-compatible-0078D6?logo=windows&logoColor=white)

Project summary
---------------
This repository implements forecasting for electrical signals: both consumption (load) and photovoltaic (solar) production. Models include classical ML and deep learning approaches to predict short-term horizons.

Problem statement and change of scope
-------------------------------------
The initial goal was to forecast the next timestep (t+15 minutes) and the next 8 timesteps (t+2 hours). During development we found data are not collected in real time but consolidated once per day (before midnight). Therefore the problem was reframed: use the previous day's consolidated data as input to predict the required horizons.

Key features
------------
- Data loaders and preprocessing for time-series (windowing, scaling).
- Multiple model families: XGBoost, KNN, MLP, GRU, CNN-GRU, Transformer.
- Training, early stopping, learning rate scheduling.
- Rolling/expanding window cross-validation and k-fold helpers.
- Prediction, denormalization and standardized metrics (MAE, MAPE, RMSE, NRMSE).
- Visualization scripts for predictions and evaluation.

Quick start
-----------
1. Clone the repository:
   git clone https://github.com/BricePetit/AIEG_Forecasting
2. Create and activate a Python 3.10+ venv:
   python -m venv .venv
   source .venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt


Project layout
--------------
- src/ : source code (data, models, training, evaluation).
- src/saved_models/ : saved model checkpoints (ignored by git).
- data/ : raw and processed datasets (ignored by git).
- notebooks/ : exploratory notebooks.
- plots/ : figures and diagnostics (ignored by git).

Usage examples
--------------
- Prepare dataset with src/data loaders or provided scripts.
- Train a model using training scripts in src/ (e.g. GRU, Transformer).
- Evaluate using expanding window or k-fold utilities.

Evaluation & metrics
--------------------
Standard metrics are provided: MAE, MAPE, MSE, RMSE, NRMSE. Predictions are denormalized before scoring when scalers are available.

Notes & recommendations
------------------------
- Because input uses daily consolidated snapshots, align modeling and operational expectations with the daily cadence.
- Consider additional exogenous features (weather forecasts, calendar) to improve day-ahead predictions.
- Use rolling/expanding validation to reflect temporal data leakage constraints.

Contributing
------------
- Open issues for bugs or feature requests.
- Submit PRs with clear descriptions and tests.
- Follow code formatting (black/flake8 recommended).

License
-------
MIT

Contact
-------
See repository metadata for authors and contact details.