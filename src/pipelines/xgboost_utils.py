"""
Shared utilities for XGBoost evaluation and inference pipelines.
"""

__title__: str = "xgboost_utils"
__version__: str = "1.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import json
import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path

# Imports from third party libraries
import pandas as pd
import xgboost as xgb
import yaml

# Imports from src
from configs.config_loader import ConfigLoader
from data.dataset_loader import build_group_data
from data.preprocessing import prepare_data_ml
from data.mysql_client import MySQLClient
from utils.site_keys import parse_domain_site_key

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------- Globals --------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
config_loader = ConfigLoader()
config = config_loader.load_global()

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Functions ------------------------------------------ #
# ----------------------------------------------------------------------------------------------- #


def build_processed_data_for_site(etype: str, site_id: int) -> pd.DataFrame:
    """
    Build processed ML dataframe for one grouped site exactly like training pipeline.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return:        Processed DataFrame ready for XGBoost inference.
    """
    grouped_sites = config['domain'][f'{etype}_sites_grouped']
    drop_periods = config['domain'][f'{etype}_drop_period']
    sites = grouped_sites[site_id]

    dataset = pd.HDFStore(f"{config['paths']['paths']['processed_data_dir']}/aieg.h5", mode='r')
    try:
        site_name, sn, prediction_type = "", "", ""
        if len(sites) > 1:
            group_data = build_group_data(dataset, sites)
            processed_data, _, _, _ = prepare_data_ml(
                site_name,
                sn,
                prediction_type,
                config["data"]["data"]["window_size"],
                config["data"]["data"]["horizon"],
                df=group_data,
                is_weather=config["data"]["features"]["use_weather_features"],
                is_ts=config["data"]["features"]["use_time_features"],
                is_stats=config["data"]["features"]["use_stat_features"],
                previous_days=config["data"]["features"]["previous_days"],
            )
        else:
            site_name, sn = parse_domain_site_key(sites[0])
            prediction_type = etype
            processed_data, _, _, _ = prepare_data_ml(
                site_name,
                sn,
                prediction_type,
                config["data"]["data"]["window_size"],
                config["data"]["data"]["horizon"],
                is_weather=config["data"]["features"]["use_weather_features"],
                is_ts=config["data"]["features"]["use_time_features"],
                is_stats=config["data"]["features"]["use_stat_features"],
                previous_days=config["data"]["features"]["previous_days"],
            )

        processed_data = processed_data.reset_index(names=['ts'])
        processed_data['site_id'] = site_id
        processed_data.set_index(['ts', 'site_id'], inplace=True)
        return clean_data_for_xgb(processed_data, {site_id: drop_periods[site_id]}, site_id)
    finally:
        dataset.close()


def build_realtime_data_for_site(
    etype: str, site_id: int
) -> pd.DataFrame:
    """
    Build processed ML dataframe for one grouped site from realtime DB data.
    This function fetches the latest data from the database instead of using the H5 file.

    :param etype:            Energy type (consumption or production).
    :param site_id:          Site ID.

    :return:                 Processed DataFrame ready for XGBoost inference.
    """
    grouped_sites = config['domain'][f'{etype}_sites_grouped']
    drop_periods = config['domain'][f'{etype}_drop_period']
    sites = grouped_sites[site_id]
    window_size = int(config["data"]["data"]["window_size"])
    previous_days = int(config["data"]["features"].get("previous_days", 0))

    # Reference timestamp for realtime inference:
    # - previous_days=0  -> now
    # - previous_days=1  -> now-24h (hier)
    reference_ts = (
        pd.Timestamp.now(tz=timezone.utc).floor("15min") - pd.Timedelta(days=previous_days)
    )

    # Load credentials and connect to database
    working_dir = config['paths']['paths']['working_dir']
    credentials_path = Path(working_dir) / "credentials.json"
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found at {credentials_path}. "
            "Cannot load realtime data from database."
        )
    
    with open(credentials_path, 'r', encoding='utf-8') as f:
        credentials = json.load(f)
    
    db = MySQLClient(
        host="192.168.0.69",
        port=3306,
        user=credentials['user'],
        password=credentials['password']
    )

    try:
        # Build raw dataframe from database
        raw_data = _load_raw_data_from_db(
            db,
            sites,
            etype,
            reference_ts=reference_ts,
            window_size=window_size,
        )
        
        if raw_data.empty:
            raise ValueError(
                f"No data available from database for site_id={site_id}, type={etype}"
            )
        
        # Apply the same preprocessing as training pipeline
        site_name, sn, prediction_type = "", "", ""
        if len(sites) > 1:
            # For grouped sites, concatenate data and process
            processed_data, _, _, _ = prepare_data_ml(
                site_name,
                sn,
                prediction_type,
                config["data"]["data"]["window_size"],
                config["data"]["data"]["horizon"],
                df=raw_data,
                is_weather=config["data"]["features"]["use_weather_features"],
                is_ts=config["data"]["features"]["use_time_features"],
                is_stats=config["data"]["features"]["use_stat_features"],
                previous_days=config["data"]["features"]["previous_days"],
            )
        else:
            site_name, sn = parse_domain_site_key(sites[0])
            prediction_type = etype
            processed_data, _, _, _ = prepare_data_ml(
                site_name,
                sn,
                prediction_type,
                config["data"]["data"]["window_size"],
                config["data"]["data"]["horizon"],
                df=raw_data,
                is_weather=config["data"]["features"]["use_weather_features"],
                is_ts=config["data"]["features"]["use_time_features"],
                is_stats=config["data"]["features"]["use_stat_features"],
                previous_days=config["data"]["features"]["previous_days"],
            )

        processed_data = processed_data.reset_index(names=['ts'])
        processed_data['site_id'] = site_id
        processed_data.set_index(['ts', 'site_id'], inplace=True)
        
        # In DB mode, the reference window is already controlled by previous_days + window_size.
        return clean_data_for_xgb(processed_data, {site_id: drop_periods[site_id]}, site_id)
    finally:
        db.connection.close()


def _load_raw_data_from_db(
    db_client: MySQLClient,
    sites: list[str],
    reference_ts: pd.Timestamp,
    window_size: int,
) -> pd.DataFrame:
    """
    Load raw data from MySQL database for the given sites.

    :param db_client:    MySQLClient instance for database queries.
    :param sites:        List of site identifiers (format: '/aieg_SITE_SN/TYPE').
    :param reference_ts: Reference timestamp used for realtime prediction.
    :param window_size:  Number of points needed in the inference window.

    :return:           Raw DataFrame with columns: ts, site, sn, ap, q1, q4 (if available).
    """
    tables_map = {
        "regie": ["std"],
        "regie_archives": [
            "std_2010", "std_2011", "std_2012", "std_2013", "std_2014", "std_2015",
            "std_2016", "std_2017", "std_2018", "std_2019", "std_2020", "std_2021",
        ],
    }

    all_dfs = []
    start_ts = reference_ts - pd.Timedelta(minutes=15 * window_size)
    start_ts_str = start_ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    end_ts_str = reference_ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")

    for site in sites:
        site_name, sn = parse_domain_site_key(site)

        # Search for data in all available tables
        for keyspace, tables in tables_map.items():
            for table in tables:
                columns = ["site", "sn", "dls", "ap", "q1", "q4"]
                table_ref = f"{keyspace}.{table}"
                
                try:
                    query = (
                        f"SELECT {', '.join(columns)} FROM {table_ref} "
                        f"WHERE site = '{site_name}' AND sn = '{sn}' "
                        f"AND dls >= '{start_ts_str}' AND dls <= '{end_ts_str}' "
                        f"ORDER BY dls ASC"
                    )
                    df = db_client.query(query)
                    
                    if df is not None and not df.empty:
                        df['dls'] = pd.to_datetime(df['dls'])
                        df = df.rename(columns={'dls': 'ts'})
                        df = df.sort_values('ts')
                        all_dfs.append(df)
                        logger.info(
                            "Loaded %d rows for %s/%s from %s (range: %s -> %s)",
                            len(df),
                            site_name,
                            sn,
                            table_ref,
                            start_ts_str,
                            end_ts_str,
                        )
                        break  # Found data for this site, stop searching
                except Exception as e:
                    logger.debug(f"Query failed for {table_ref}/{site_name}/{sn}: {e}")
                    continue

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['ts', 'site', 'sn'])
    else:
        return pd.DataFrame()


def load_selected_features(etype: str, site_id: int) -> list[str]:
    """
    Load selected features saved during training.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.

    :return: List of selected feature names.
    """
    feature_path = Path(f"src/configs/features/{etype}/site_{site_id}.yaml")
    if not feature_path.exists():
        raise FileNotFoundError(f"Selected features file not found: {feature_path}")
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_cfg = yaml.safe_load(f) or {}
    selected_features = feature_cfg.get("selected_features", [])
    if not selected_features:
        raise ValueError(f"No selected_features found in {feature_path}")
    return selected_features


def load_xgb_model(etype: str, site_id: int) -> xgb.Booster:
    """
    Load XGBoost model saved by training pipeline.

    :param etype:   Energy type (consumption or production).
    :param site_id: Site ID.
        
    :return: Loaded XGBoost Booster model.
    """
    model_path = (
        Path(config['paths']['paths']['saved_models_dir']) / f"xgb_{etype}_siteid_{site_id}.json"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def clean_data_for_xgb(
    processed_data: pd.DataFrame, period2drop: dict, site_id: int
) -> pd.DataFrame:
    """
    Drop invalid periods from processed data for a given site.

    :param processed_data:  DataFrame with processed data for the site.
    :param period2drop:     Dictionary mapping site_id to list of (start, end) periods to drop.
    :param site_id:         ID of the site for which to clean data.

    :return: Cleaned DataFrame with invalid periods removed.
    """
    if site_id in period2drop and len(period2drop[site_id][0]) > 0:
        for period in period2drop[site_id]:
            start, end = period
            if start == "start":
                start = processed_data.index.get_level_values("ts").min()
            if end == "end":
                end = processed_data.index.get_level_values("ts").max()
            processed_data = processed_data.drop(
                processed_data.loc[
                    (processed_data.index.get_level_values("ts") >= start)
                    & (processed_data.index.get_level_values("ts") <= end)
                ].index
            )
    return processed_data


def save_predictions(
    site_id: int, predictions_df: pd.DataFrame, etype: str, config_dir: str = "src"
) -> str:
    """
    Save predictions to YAML file following same structure as save_best_params.

    :param site_id:         ID of the site for which predictions were generated.
    :param predictions_df:  DataFrame containing predictions.
    :param etype:           Type of prediction (consumption or production).
    :param config_dir:      Directory where predictions will be saved.

    :return: Path to saved predictions file.
    """
    predictions_root = config['paths']['paths'].get('predictions_dir')
    if predictions_root:
        path = Path(predictions_root) / etype / f"site_{site_id}.yaml"
    else:
        path = Path(config_dir) / f"predictions/{etype}/site_{site_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataframe to a YAML-friendly structure:
    # - pandas Timestamps -> ISO strings
    # - numpy scalars -> native Python types
    serializable_df = predictions_df.reset_index().copy()
    for col in serializable_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable_df[col]):
            serializable_df[col] = serializable_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    predictions_data = serializable_df.to_dict(orient='records')
    for record in predictions_data:
        for key, value in list(record.items()):
            if hasattr(value, "item"):
                record[key] = value.item()

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(predictions_data, f, sort_keys=False, default_flow_style=False)

    logger.info("Predictions saved to %s", path)
    return str(path)


def save_predictions_history(
    site_id: int,
    predictions_df: pd.DataFrame,
    etype: str,
    run_id: str,
    data_source: str,
    inference_mode: str,
    created_at_utc: str | None = None,
    config_dir: str = "src",
) -> str:
    """
    Save predictions history as append-only parquet artifact for traceability.

    :param site_id:         ID of the site for which predictions were generated.
    :param predictions_df:  DataFrame containing predictions.
    :param etype:           Type of prediction (consumption or production).
    :param run_id:          Unique run identifier.
    :param data_source:     Data source used for inference (h5 or db).
    :param inference_mode:  Inference mode (next_step or batch).
    :param created_at_utc:  Run timestamp in UTC ISO format. If None, generated now.
    :param config_dir:      Directory where fallback predictions root can be created.

    :return: Path to saved history parquet file.
    """
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).isoformat()

    predictions_root = config['paths']['paths'].get('predictions_dir')
    if predictions_root:
        history_root = Path(predictions_root) / "history" / etype / f"site_{site_id}"
    else:
        history_root = Path(config_dir) / "predictions/history" / etype / f"site_{site_id}"

    date_partition = created_at_utc[:10]
    run_dir = history_root / f"date={date_partition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.parquet"

    serializable_df = predictions_df.reset_index().copy()
    pred_cols = [c for c in serializable_df.columns if c.startswith("pred_t+")]
    if not pred_cols:
        raise ValueError("No prediction columns found (expected pred_t+*)")

    rows: list[pd.DataFrame] = []
    for pred_col in pred_cols:
        try:
            horizon_step = int(pred_col.split("+")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid prediction column format: {pred_col}") from exc

        frame = serializable_df[["ts", "site_id", pred_col]].copy()
        frame = frame.rename(columns={pred_col: "y_pred"})
        frame["horizon_step"] = horizon_step
        frame["target_ts"] = pd.to_datetime(frame["ts"]) + pd.to_timedelta(
            frame["horizon_step"] * 15, unit="min"
        )

        true_col = f"true_ap+{horizon_step}"
        frame["y_true"] = serializable_df[true_col] if true_col in serializable_df.columns else pd.NA
        rows.append(frame)

    history_df = pd.concat(rows, ignore_index=True)
    history_df["run_id"] = run_id
    history_df["created_at_utc"] = created_at_utc
    history_df["etype"] = etype
    history_df["site_id"] = site_id
    history_df["data_source"] = data_source
    history_df["inference_mode"] = inference_mode

    history_df = history_df[[
        "run_id",
        "created_at_utc",
        "etype",
        "site_id",
        "data_source",
        "inference_mode",
        "ts",
        "target_ts",
        "horizon_step",
        "y_pred",
        "y_true",
    ]]

    history_df.to_parquet(output_path, index=False)
    logger.info("Predictions history saved to %s", output_path)
    return str(output_path)
