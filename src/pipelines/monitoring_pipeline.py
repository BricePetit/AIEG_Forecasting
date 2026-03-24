"""
Monitoring pipeline for production predictions.

Monitors model performance in production by comparing predictions with actual values
as they arrive. Tracks metrics over time and detects performance degradation/drift.
"""

__title__: str = "monitoring_pipeline"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import logging
from pathlib import Path
from typing import Any

# Imports from third party libraries
import pandas as pd
import numpy as np
import yaml

# Imports from src
from configs.config_loader import ConfigLoader
from evaluation.metrics import compute_metrics
from utils.logging import setup_logger

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="monitoring_pipeline.log", level=logging.INFO)

config_loader = ConfigLoader()
config = config_loader.load_global()

DASH = '-' * 20

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def load_actual_values(etype: str, site_id: int) -> pd.DataFrame | None:
    """
    Load actual observed values (should come from database or sensor updates).
        
    Placeholder - in production, this would query the database for recent observations.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return:        DataFrame with actual values from H5 or None if not available.
    """
    try:
        h5_path = f"{config['paths']['paths']['processed_data_dir']}/aieg.h5"
        dataset = pd.HDFStore(h5_path, mode='r')

        grouped_sites = config['domain'][f'{etype}_sites_grouped']
        if site_id not in grouped_sites:
            logger.warning("Site %s not found in %s sites_grouped", site_id, etype)
            return None

        sites = grouped_sites[site_id]

        # For simplicity, load the most recent data from H5
        # In production, this should query live data
        if len(sites) == 1:
            site_key = sites[0]
            table_name = (
                f"/aieg_{site_key.split('_')[1]}_{site_key.split('_')[2].split('/')[0]}/{etype}"
            )
            try:
                df = dataset[table_name].set_index('ts')[['ap']]
                # Get most recent data (last 96 values = 1 day at 15min intervals)
                df = df.tail(96)
                dataset.close()
                return df
            except KeyError:
                logger.warning("Table %s not found in H5", table_name)
                dataset.close()
                return None
        else:
            logger.warning("Multi-site aggregation not implemented for monitoring yet")
            dataset.close()
            return None

    except Exception as e:
        logger.exception("Failed to load actual values for site %s: %s", site_id, e)
        return None


def load_predictions(etype: str, site_id: int) -> pd.DataFrame | None:
    """
    Load predictions saved by inference pipeline.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return:        DataFrame with predictions or None if not found.
    """
    prediction_path = Path(f"src/configs/predictions/{etype}/site_{site_id}.yaml")
    if not prediction_path.exists():
        logger.warning("Prediction not found: %s", prediction_path)
        return None

    with open(prediction_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []

    if not data:
        return None

    df = pd.DataFrame(data)
    return df


def compute_model_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard metrics for model monitoring.

    :param y_pred:  Predicted values.
    :param y_true:  True/actual values.

    :return:        Dictionary with metrics.
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return {}

    # Reshape for compatibility with compute_metrics
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    try:
        metrics = compute_metrics(y_true, y_pred)
        return metrics
    except Exception as e:
        logger.warning("Could not compute metrics: %s", e)
        return {}


def monitor_site(etype: str, site_id: int) -> dict[str, Any]:
    """
    Monitor performance for one site by comparing predictions with actual values.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return:        Monitoring report with metrics and alerts.
    """
    logger.info("%s Monitoring | type=%s | site_id=%s %s", DASH, etype, site_id, DASH)

    predictions_df = load_predictions(etype, site_id)
    actual_df = load_actual_values(etype, site_id)

    if predictions_df is None:
        return {"status": "error", "message": "No predictions available"}

    if actual_df is None:
        return {"status": "no_data", "message": "No actual values available yet for monitoring"}

    # Align on index
    if 'ts' in predictions_df.columns:
        predictions_df = predictions_df.set_index('ts')
    if isinstance(actual_df.index, str):
        actual_df.index = pd.to_datetime(actual_df.index)

    # Find matching indices
    common_idx = predictions_df.index.intersection(actual_df.index)

    if len(common_idx) == 0:
        logger.warning("No overlapping timestamps between predictions and actuals for site %s", site_id)
        return {"status": "no_overlap", "message": "No timestamp overlap between predictions and actuals"}

    y_pred = predictions_df.loc[common_idx, "pred_t+1"].values
    y_true = actual_df.loc[common_idx, "ap"].values

    metrics = compute_model_metrics(y_pred, y_true)

    # Check for degradation (simple threshold-based)
    mape = metrics.get("mape", np.inf)
    degradation_alert = False
    if mape > 20.0:  # Example threshold: MAPE > 20%
        degradation_alert = True
        logger.warning("DEGRADATION WARNING: MAPE=%.2f%% for site %s", mape, site_id)

    return {
        "status": "success",
        "type": etype,
        "site_id": site_id,
        "n_comparisons": len(common_idx),
        "metrics": metrics,
        "degradation_alert": degradation_alert,
        "last_update": pd.Timestamp.now().isoformat(),
    }


def run_monitoring_pipeline(
    etype: str | None = None,
    site_ids: list[int] | None = None,
    save_report: bool = True,
) -> dict[str, Any]:
    """
    Run monitoring across all or specified sites.

    :param etype:       Energy type. If None, monitor both.
    :param site_ids:    List of specific site IDs. If None, monitor all.
    :param save_report: Save monitoring report to YAML.

    :return:            Monitoring report dictionary.
    """
    logger.info("%s Starting monitoring pipeline %s", DASH, DASH)

    energy_types = [etype] if etype else ["consumption", "production"]
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "sites": {},
        "degradation_alerts": [],
        "summary": {},
    }

    for current_type in energy_types:
        current_group = config['domain'][f'{current_type}_sites_grouped']
        current_site_ids = site_ids if site_ids else list(current_group.keys())
        report["sites"][current_type] = {}

        for site_id in current_site_ids:
            result = monitor_site(current_type, site_id)
            report["sites"][current_type][site_id] = result

            if result.get("degradation_alert"):
                report["degradation_alerts"].append({
                    "type": current_type,
                    "site_id": site_id,
                    "metrics": result.get("metrics", {})
                })

        report["summary"][current_type] = {
            "n_sites_monitored": len(current_site_ids),
            "n_sites_with_data": sum(1 for r in report["sites"][current_type].values() if r["status"] == "success"),
            "n_degradation_alerts": sum(1 for r in report["sites"][current_type].values() if r.get("degradation_alert")),
        }

    # Save report
    if save_report:
        output_path = Path("src/configs/monitoring_reports")
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / f"monitoring_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(report_file, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
        logger.info("Monitoring report saved to %s", report_file)
        report["output_path"] = str(report_file)

    logger.info("%s Monitoring pipeline completed %s", DASH, DASH)
    return report


if __name__ == "__main__":
    run_monitoring_pipeline()
