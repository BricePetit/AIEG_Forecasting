"""
Inference pipeline for the forecasting task.
"""

__title__: str = "inference_pipeline"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

# Imports from third party libraries
import pandas as pd
import xgboost as xgb

# Imports from src
from configs.config_loader import ConfigLoader
from data.weather_loader import download_irm_data
from utils.logging import setup_logger
from xgboost_utils import (
    build_processed_data_for_site,
    build_realtime_data_for_site,
    load_selected_features,
    load_xgb_model,
    save_predictions_history,
    save_predictions,
)

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="inference_pipeline.log", level=logging.INFO)

config_loader = ConfigLoader()
config = config_loader.load_global()

DASH = '-' * 20

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def infer_xgb_site(
    etype: str,
    site_id: int,
    use_test_split: bool = False,
    next_step_only: bool = True,
    data_delay_days: int = 1,
    data_source: str = "h5",
) -> pd.DataFrame:
    """
    Run inference for one site group and return predictions dataframe.

    :param etype:           Type of energy (consumption or production).
    :param site_id:         Site ID.
    :param use_test_split:  Use only test split (last 10%) for inference.
    :param next_step_only:  If True, infer only one next point per site.
    :param data_delay_days: Delay in days for data availability (e.g., 1 for J-1 mode).
    :param data_source:     Data source: "h5" for test data (H5 file) or "db" for realtime (database).

    :return:                DataFrame with predictions and optionally true values.
    """
    logger.info(
        "%s Running inference | type=%s | site_id=%s | source=%s %s",
        DASH, etype, site_id, data_source, DASH
    )
    
    # Load data from appropriate source
    if data_source.lower() == "db":
        # Realtime mode: load from database
        processed_data = build_realtime_data_for_site(etype, site_id, data_delay_days)
    elif data_source.lower() == "h5":
        # Test mode: load from H5 file (training dataset)
        processed_data = build_processed_data_for_site(etype, site_id)
    else:
        raise ValueError(
            f"Invalid data_source: {data_source}. Must be 'h5' or 'db'."
        )
    
    selected_features = load_selected_features(etype, site_id)
    model = load_xgb_model(etype, site_id)

    target_cols = [c for c in processed_data.columns if c.startswith("ap+")]
    missing = [c for c in selected_features if c not in processed_data.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} selected features in inference data for site {site_id}: {missing[:5]}"
        )

    if next_step_only:
        # Operational mode: use data available with delay (typically J-1),
        # then predict only the next point from the latest eligible window.
        if data_source.lower() == "h5":
            # For H5 source, apply cutoff filtering
            cutoff_ts = pd.Timestamp.now().floor("15min") - pd.Timedelta(days=data_delay_days)
            eligible_data = processed_data.loc[
                processed_data.index.get_level_values("ts") <= cutoff_ts
            ]
            if eligible_data.empty:
                raise ValueError(
                    "No eligible rows found for delayed inference "
                    f"(cutoff_ts={cutoff_ts}, delay_days={data_delay_days})"
                )
            data_for_inference = eligible_data.tail(1)
        else:
            # For DB source, data is already filtered by build_realtime_data_for_site
            data_for_inference = processed_data.tail(1)
    else:
        data_for_inference = (
            processed_data.iloc[int(len(processed_data) * 0.9):]
            if use_test_split else processed_data
        )
    if data_for_inference.empty:
        raise ValueError(f"No rows available for inference for type={etype}, site_id={site_id}")

    dmatrix = xgb.DMatrix(data_for_inference[selected_features])
    y_pred = model.predict(dmatrix)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    pred_cols = [f"pred_t+{i+1}" for i in range(y_pred.shape[1])]
    predictions_df = pd.DataFrame(y_pred, index=data_for_inference.index, columns=pred_cols)

    # Add true targets if available, useful for quick diagnostics.
    for col in target_cols:
        if col in data_for_inference.columns:
            predictions_df[f"true_{col}"] = data_for_inference[col].values

    logger.info(
        "Inference completed for type=%s site_id=%s (source=%s) with %d rows",
        etype,
        site_id,
        data_source,
        len(predictions_df),
    )
    return predictions_df


def run_inference_pipeline(
    etype: str | None = None,
    site_ids: list[int] | None = None,
    use_test_split: bool = False,
    next_step_only: bool = True,
    data_delay_days: int = 1,
    data_source: str = "h5",
    refresh_irm_weather: bool = True,
    save_pred: bool = True,
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Run inference for all site groups and optionally save predictions to YAML.

    :param etype:               Type of energy (consumption, production). If None, run for both.
    :param site_ids:            List of specific site IDs to process. If None, process all.
    :param use_test_split:      Use only test split (last 10%) for inference.
    :param next_step_only:      If True, infer only one next point per site.
    :param data_delay_days:     Delay in days for data availability (e.g., 1 for J-1 mode).
    :param data_source:         Data source: "h5" for test data (H5 file) or "db" for realtime (database).
    :param refresh_irm_weather: If True, refresh IRM weather file before inference.
    :param save_pred:           Save both latest snapshot (YAML) and history (Parquet).

    :return:                    Dictionary with results for each type and site_id.
    """
    if config['model']['model']['cnn_gru']['selected']:
        raise NotImplementedError(
            "Current inference pipeline implementation targets XGBoost artifacts. "
            "Set cnn_gru.selected=false or extend this pipeline for CNN-GRU artifacts."
        )

    energy_types = [etype] if etype else ["consumption", "production"]
    results: dict[str, dict[int, dict[str, Any]]] = {}
    run_started_at_utc = datetime.now(timezone.utc)
    run_id = run_started_at_utc.strftime("%Y%m%dT%H%M%SZ") + f"_{uuid4().hex[:8]}"
    run_created_at_utc = run_started_at_utc.isoformat()
    logger.info("%s Starting inference pipeline %s", DASH, DASH)

    if refresh_irm_weather and config["data"]["features"].get("use_weather_features", False):
        now_year = datetime.now().year
        output_file = f"{config['paths']['paths']['processed_data_dir']}/aws_10min.csv"
        start_year = now_year - 1

        weather_path = Path(output_file)
        if weather_path.exists():
            try:
                weather_index = pd.read_csv(output_file, usecols=["timestamp"])
                weather_index["timestamp"] = pd.to_datetime(
                    weather_index["timestamp"], errors="coerce"
                )
                weather_index = weather_index.dropna(subset=["timestamp"])
                if not weather_index.empty:
                    start_year = int(weather_index["timestamp"].min().year)
            except Exception as exc:
                logger.warning(
                    "Could not infer start year from existing weather file (%s). "
                    "Fallback to %d.",
                    exc,
                    start_year,
                )

        logger.info(
            "Refreshing IRM weather file for inference from %d to %d: %s",
            start_year,
            now_year,
            output_file,
        )
        download_irm_data(output_file=output_file, start_year=start_year, end_year=now_year)

    for current_type in energy_types:
        current_group = config['domain'][f'{current_type}_sites_grouped']
        current_site_ids = site_ids if site_ids else list(current_group.keys())
        results[current_type] = {}

        for site_id in current_site_ids:
            try:
                predictions_df = infer_xgb_site(
                    current_type,
                    site_id,
                    use_test_split=use_test_split,
                    next_step_only=next_step_only,
                    data_delay_days=data_delay_days,
                    data_source=data_source,
                )
                output_path = None
                history_output_path = None
                if save_pred:
                    output_path = save_predictions(site_id, predictions_df, current_type)
                    history_output_path = save_predictions_history(
                        site_id=site_id,
                        predictions_df=predictions_df,
                        etype=current_type,
                        run_id=run_id,
                        data_source=data_source,
                        inference_mode="next_step" if next_step_only else "batch",
                        created_at_utc=run_created_at_utc,
                    )

                results[current_type][site_id] = {
                    "type": current_type,
                    "site_id": site_id,
                    "n_rows": len(predictions_df),
                    "inference_mode": "next_step" if next_step_only else "batch",
                    "data_source": data_source,
                    "run_id": run_id,
                    "history_output_path": history_output_path,
                    "output_path": output_path,
                }
            except Exception as exc:
                logger.exception("Inference failed for type=%s site_id=%s: %s", current_type, site_id, exc)
                results[current_type][site_id] = {"error": str(exc)}

    logger.info("%s Inference pipeline completed %s", DASH, DASH)
    return results


if __name__ == "__main__":
    # Example usage:
    # 
    # Mode 1: Test avec données H5 (défaut)
    # run_inference_pipeline(data_source='h5')
    #
    # Mode 2: Temps réel avec données DB (J-1)
    # run_inference_pipeline(data_source='db', data_delay_days=1)
    #
    # Mode 3: Temps réel avec sites spécifiques
    # run_inference_pipeline(
    #     etype='consumption',
    #     site_ids=[1, 2],
    #     data_source='db',
    #     next_step_only=True,
    #     data_delay_days=1
    # )
    run_inference_pipeline()
