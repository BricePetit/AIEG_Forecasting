"""
Evaluation pipeline for the forecasting task.
"""

__title__: str = "evaluation_pipeline"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import logging
from typing import Any

# Imports from third party libraries
import xgboost as xgb

# Imports from src
from configs.config_loader import ConfigLoader
from evaluation.metrics import compute_metrics, print_metrics
from utils.logging import setup_logger
from xgboost_utils import build_processed_data_for_site, load_selected_features, load_xgb_model

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="evaluation_pipeline.log", level=logging.INFO)

config_loader = ConfigLoader()
config = config_loader.load_global()

DASH = '-' * 20

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def evaluate_xgb_site(etype: str, site_id: int) -> dict[str, Any]:
    """
    Evaluate one XGBoost model on the test split.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return:        Dictionary with evaluation results and metrics.
    """
    site_group = config['domain'][f'{etype}_sites_grouped'].get(site_id, [])
    site_label = ", ".join(site_group) if site_group else "unknown"
    logger.info(
        "%s Evaluating XGBoost | type=%s | site_id=%s | sites=%s %s",
        DASH,
        etype,
        site_id,
        site_label,
        DASH,
    )
    processed_data = build_processed_data_for_site(etype, site_id)
    selected_features = load_selected_features(etype, site_id)
    model = load_xgb_model(etype, site_id)

    target_cols = [c for c in processed_data.columns if c.startswith("ap+")]
    if not target_cols:
        raise ValueError("No target columns found (expected columns starting with 'ap+')")

    test = processed_data.iloc[int(len(processed_data) * 0.9):]
    if test.empty:
        raise ValueError(f"Empty test set for type={etype}, site_id={site_id}")

    missing = [c for c in selected_features if c not in test.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} selected features in evaluation data for site "
            f"{site_id}: {missing[:5]}"
        )

    dtest = xgb.DMatrix(test[selected_features], label=test[target_cols])
    y_pred = model.predict(dtest)
    y_true = test[target_cols].values

    print_metrics(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(
        "Computed metrics for type=%s site_id=%s sites=%s: %s",
        etype,
        site_id,
        site_label,
        metrics,
    )

    return {
        "type": etype,
        "site_id": site_id,
        "n_rows_test": len(test),
        "metrics": metrics,
    }


def run_evaluation_pipeline(
    etype: str | None = None, site_ids: list[int] | None = None
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Run evaluation for one or many site groups.

    :param etype:       Energy type. If None, evaluate both.
    :param site_ids:    List of specific site IDs. If None, evaluate all.

    :return:            Nested dictionary with evaluation results for each type and site.
    """
    if config['model']['model']['cnn_gru']['selected']:
        raise NotImplementedError(
            "Current evaluation pipeline implementation targets XGBoost artifacts. "
            "Set cnn_gru.selected=false or extend this pipeline for CNN-GRU artifacts."
        )

    energy_types = [etype] if etype else ["consumption", "production"]
    results: dict[str, dict[int, dict[str, Any]]] = {}
    logger.info("%s Starting evaluation pipeline %s", DASH, DASH)

    for current_type in energy_types:
        current_group = config['domain'][f'{current_type}_sites_grouped']
        current_site_ids = site_ids if site_ids else list(current_group.keys())
        results[current_type] = {}
        for site_id in current_site_ids:
            site_group = current_group.get(site_id, [])
            site_label = ", ".join(site_group) if site_group else "unknown"
            try:
                results[current_type][site_id] = evaluate_xgb_site(current_type, site_id)
            except Exception as exc:
                logger.exception(
                    "Evaluation failed for type=%s site_id=%s sites=%s: %s",
                    current_type,
                    site_id,
                    site_label,
                    exc,
                )
                results[current_type][site_id] = {"error": str(exc)}

    logger.info("%s Evaluation pipeline completed %s", DASH, DASH)
    return results


if __name__ == "__main__":
    run_evaluation_pipeline()
