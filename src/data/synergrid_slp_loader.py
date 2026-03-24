"""
Minimal Synergrid loader for electricity RLP all-DSO files.

Design choices:
- Keep only one path: RLP electricity (no SLP EX, no gas).
- Parse only the `RLP96UbyDGO` sheet.
- Extract one DSO curve: target DSO (default AIEG), fallback SMALL family.
- Keep API compatibility with previous pipeline call signature.
"""

__title__: str = "synergrid_slp_loader"
__version__: str = "2.0.0"
__author__: str = "Brice Petit"
__license__: str = "MIT"

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- IMPORTS ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Imports from standard library
import logging
import re
from urllib.parse import unquote
from datetime import datetime, timezone
from pathlib import Path

# Imports from third party libraries
import pandas as pd
import requests

# Imports from src
from utils.logging import setup_logger

# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------- Globals ------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    setup_logger(log_file="synergrid_slp_loader.log", level=logging.INFO)

DASH = "-" * 20
SYNERGRID_SLP_URL = (
    "https://www.synergrid.be/fr/centre-de-documentation/statistiques-et-donnees/profils-slp-spp-rlp"
)
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-BE,fr;q=0.9,en;q=0.8",
    "Referer": "https://www.synergrid.be/",
}

# Stable direct links for recent years.
RLP_ELECTRICITY_URL_OVERRIDES: dict[int, str] = {
    2016: "https://www.synergrid.be/images/downloads/rlp0n2016-electricity-excl-kcf.xls",
    2017: "https://www.synergrid.be/images/downloads/rlp0n2017-electricity-excl-kcf.xls",
    2018: "https://www.synergrid.be/images/downloads/rlp0n2018-electricity-excl-kcf.xls",
    2019: "https://www.synergrid.be/images/downloads/rlp0n2019-electricity-excl-kcf.xls",
    2020: "https://www.synergrid.be/images/downloads/rlp0n2020-electricity-excl-kcf.xls",
    2021: "https://www.synergrid.be/images/downloads/rlp0n2021-electricity-excl-kcf.xls",
    2022: "https://www.synergrid.be/images/downloads/rlp0n2022-electricity-all-dsos.xlsb",
    2023: "https://www.synergrid.be/images/downloads/rlp0n2023-electricity-all-dsos.xlsb",
    2024: "https://www.synergrid.be/images/downloads/profielen_website/rlp0n2024_elec_all_DSOs.xlsb",
    2025: "https://www.synergrid.be/images/downloads/RLP0N%202025%20Electricity%20all%20DSOs.xlsb",
    2026: "https://www.synergrid.be/images/downloads/SLP-RLP-SPP/2026/RLP0N%202026%20Electricity%20all%20DSOs.xlsb",
}


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Helpers -------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def _extract_anchor_links(html: str) -> list[tuple[str, str]]:
    """
    Extract (label, url) anchors from page HTML.
    """
    pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    links: list[tuple[str, str]] = []
    for href, raw_label in pattern.findall(html):
        label = re.sub(r"<[^>]+>", "", raw_label)
        label = re.sub(r"\s+", " ", label).strip()
        if not label:
            continue
        url = href if href.startswith("http") else f"https://www.synergrid.be{href}"
        links.append((label, url))
    return links


def _discover_rlp_links(start_year: int, end_year: int) -> dict[int, str]:
    """
    Discover RLP electricity links by year.

    The selection is intentionally simple:
    - use hardcoded overrides when available,
    - else pick first page link containing year + rlp + elec/electricity and excluding gas.

    :param start_year: Start year for link discovery.
    :param end_year:   End year for link discovery.

    :return:           Dict mapping year to discovered URL.
    """
    response = requests.get(SYNERGRID_SLP_URL, headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()
    links = _extract_anchor_links(response.text)

    out: dict[int, str] = {}
    for year in range(start_year, end_year + 1):
        if year in RLP_ELECTRICITY_URL_OVERRIDES:
            out[year] = RLP_ELECTRICITY_URL_OVERRIDES[year]
            continue

        y = str(year)
        for label, url in links:
            txt = unquote(f"{label} {url}".lower())
            if not re.search(rf"(?<!\\d){y}(?!\\d)", txt):
                continue
            if "rlp" not in txt:
                continue
            if "elec" not in txt and "electricity" not in txt:
                continue
            if "gas" in txt or "parameter" in txt:
                continue
            out[year] = url
            break

    return out


def _select_dso_column(df: pd.DataFrame, dso_target: str, small_dso_fallback: str) -> str:
    """
    Pick the DSO column from `RLP96UbyDGO`.

    Priority:
    1) exact column name == dso_target
    2) metadata row (first rows) contains dso_target label
    3) exact fallback column (e.g. SMALL)
    4) first fallback family variant (e.g. SMALL.1)

    :param df:                 DataFrame with DSO columns.
    :param dso_target:         Target DSO name to look for (e.g., AIEG).
    :param small_dso_fallback: Fallback DSO family name (e.g., SMALL) if target not found.

    :return:                    Selected column name.

    :raises ValueError:         If no suitable column found.
    """
    target_l = dso_target.strip().lower()
    fallback_l = small_dso_fallback.strip().lower()
    cols = [str(c).strip() for c in df.columns]

    for col in cols:
        if col.lower() == target_l:
            return col

    for col in cols:
        head_vals = df[col].head(4).astype(str).str.strip().str.lower().tolist()
        if target_l in head_vals:
            return col

    for col in cols:
        if col.lower() == fallback_l:
            return col

    family = [c for c in cols if c.lower().startswith(fallback_l)]
    if family:
        variants = [c for c in family if c.lower() != fallback_l]
        return variants[0] if variants else family[0]

    raise ValueError(f"No suitable DSO column found for target '{dso_target}'")


def _read_rlp_all_dso_profile(
    file_path: Path, dso_target: str, small_dso_fallback: str
) -> pd.DataFrame:
    """
    Parse one RLP all-DSO workbook into `ts` + `value`.

    :param file_path:          Path to the downloaded RLP Excel file.
    :param dso_target:         Target DSO name to look for (e.g.,
                                typically "AIEG" for AIEG forecasting).
    :param small_dso_fallback: Fallback DSO family name (e.g., "SMALL") if target not found.

    :return:                    DataFrame with `ts` and `value` columns.
    """
    xls = pd.ExcelFile(file_path)
    if "RLP96UbyDGO" not in xls.sheet_names:
        raise ValueError(f"RLP96UbyDGO not found in {file_path}")

    raw = pd.read_excel(file_path, sheet_name="RLP96UbyDGO", header=None)
    if raw.empty:
        raise ValueError(f"RLP96UbyDGO empty in {file_path}")

    # Find the row carrying time headers CET/Year/Month/Day/h/Min.
    header_row = None
    for i in range(min(10, len(raw))):
        vals = raw.iloc[i, :6].astype(str).str.strip().str.lower().tolist()
        if vals == ["cet", "year", "month", "day", "h", "min"]:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not locate CET/Year/Month/Day/h/Min header row")

    dgo_row = max(header_row - 1, 0)
    columns: list[str] = []
    used: dict[str, int] = {}
    for idx in range(raw.shape[1]):
        if idx < 7:
            name = str(raw.iloc[header_row, idx]).strip()
        else:
            name = str(raw.iloc[dgo_row, idx]).strip()
            if name in {"", "nan", "None"}:
                name = str(raw.iloc[header_row, idx]).strip()
        if name in {"", "nan", "None"}:
            name = f"col_{idx}"

        base = name
        n = used.get(base, 0)
        if n > 0:
            name = f"{base}.{n}"
        used[base] = n + 1
        columns.append(name)

    df = raw.iloc[header_row + 1 :].copy()
    df.columns = columns

    col_map = {str(c).strip().lower(): c for c in df.columns}
    ts = pd.Series(pd.NaT, index=df.index)
    cet_col = col_map.get("cet")
    if cet_col is not None:
        cet_num = pd.to_numeric(df[cet_col], errors="coerce")
        ts = pd.to_datetime(cet_num, unit="D", origin="1899-12-30", errors="coerce")

    if float(ts.notna().mean()) < 0.5:
        y_col = col_map.get("year")
        m_col = col_map.get("month")
        d_col = col_map.get("day")
        h_col = col_map.get("h")
        min_col = col_map.get("min")
        if all(c is not None for c in [y_col, m_col, d_col, h_col, min_col]):
            ts = pd.to_datetime(
                {
                    "year": pd.to_numeric(df[y_col], errors="coerce"),
                    "month": pd.to_numeric(df[m_col], errors="coerce"),
                    "day": pd.to_numeric(df[d_col], errors="coerce"),
                    "hour": pd.to_numeric(df[h_col], errors="coerce"),
                    "minute": pd.to_numeric(df[min_col], errors="coerce"),
                },
                errors="coerce",
            )

    value_col = _select_dso_column(
        df, dso_target=dso_target, small_dso_fallback=small_dso_fallback
    )
    values = pd.to_numeric(df[value_col], errors="coerce")

    out = pd.DataFrame({"ts": ts, "value": values}).dropna(subset=["ts", "value"])
    out = out[(out["ts"].dt.year >= 2000) & (out["ts"].dt.year <= 2100)]
    out = out.drop_duplicates(subset=["ts"]).sort_values("ts")
    if len(out) < 20_000:
        raise ValueError(f"Too few points parsed in {file_path}: {len(out)}")
    return out.reset_index(drop=True)


def _to_watts_from_profile(
    profile_df: pd.DataFrame,
    annual_consumption_kwh: float,
    resolution_minutes: int = 15,
) -> pd.DataFrame:
    """
    Convert RLP values to watts.

    Handles normalized curves (sum ~1 or ~100) and direct power values.

    :param profile_df:              DataFrame with `ts` and `value` columns from RLP profile.
    :param annual_consumption_kwh:  Annual consumption in kWh to scale the profile.
    :param resolution_minutes:      Time resolution of the profile in minutes (default 15).

    :return:                        DataFrame with `ts` and `power_w` columns.
    """
    out = profile_df.copy()
    values = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)

    v_sum = float(values.sum())
    v_max = float(values.max())

    if 0.8 <= v_sum <= 1.2:
        energy_kwh = annual_consumption_kwh * values
        out["power_w"] = energy_kwh / (resolution_minutes / 60.0) * 1000.0
    elif 80.0 <= v_sum <= 120.0:
        shares = values / 100.0
        energy_kwh = annual_consumption_kwh * shares
        out["power_w"] = energy_kwh / (resolution_minutes / 60.0) * 1000.0
    elif v_max < 50:
        out["power_w"] = values * 1000.0
    else:
        out["power_w"] = values

    return out[["ts", "power_w"]]


# ----------------------------------------------------------------------------------------------- #
# ------------------------------------------ Public API ----------------------------------------- #
# ----------------------------------------------------------------------------------------------- #


def download_and_build_synergrid_consumption_csv(
    output_csv: str,
    raw_dir: str,
    start_year: int = 2010,
    end_year: int | None = None,
    annual_consumption_kwh: float = 3500.0,
    resolution_minutes: int = 15,
    profile_kind: str = "rlp0n",
    rlp_start_year: int = 2016,
    dso_target: str = "AIEG",
    small_dso_fallback: str = "SMALL",
) -> str:
    """
    Download Synergrid files and build one synthetic consumption curve CSV (watts).

    Kept parameters for compatibility. This implementation is RLP electricity only.

    :param output_csv:              Path to output CSV file.
    :param raw_dir:                 Directory to store downloaded raw files.
    :param start_year:              Start year for data (default 2010).
    :param end_year:                End year for data (default current year).
    :param annual_consumption_kwh:  Annual consumption in kWh to scale the profile
                                    (default 3500 kWh).
    :param resolution_minutes:      Time resolution of the profile in minutes (default 15).
    :param profile_kind:            Profile kind to load (default "rlp0n", only supported option).
    :param rlp_start_year:          Earlyest year for RLP data (default 2016, as earlier years have
                                    more parsing issues).
    :param dso_target:              Target DSO name to look for in RLP files (default "AIEG").
    :param small_dso_fallback:      Fallback DSO family name to look for if target not found
                                    (default "SMALL").

    :return:                        Path to the output CSV file.
    """
    if profile_kind != "rlp0n":
        raise ValueError("This simplified loader supports only profile_kind='rlp0n'")

    if end_year is None:
        end_year = datetime.now(timezone.utc).year

    start_year = max(start_year, rlp_start_year)

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    links = _discover_rlp_links(start_year=start_year, end_year=end_year)
    if not links:
        raise ValueError("No RLP electricity links discovered for requested years")

    yearly_curves: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        url = links.get(year)
        if not url:
            logger.warning("No RLP electricity link found for year %d", year)
            continue

        ext = Path(url).suffix or ".xls"
        file_path = raw_path / f"synergrid_slp_{year}{ext}"

        logger.info("%s Downloading RLP file for %d: %s %s", DASH, year, url, DASH)
        response = requests.get(url, headers=HTTP_HEADERS, timeout=120)
        response.raise_for_status()
        file_path.write_bytes(response.content)

        try:
            profile_df = _read_rlp_all_dso_profile(
                file_path,
                dso_target=dso_target,
                small_dso_fallback=small_dso_fallback,
            )
            curve_w = _to_watts_from_profile(
                profile_df,
                annual_consumption_kwh=annual_consumption_kwh,
                resolution_minutes=resolution_minutes,
            )
            curve_w["year"] = year
            yearly_curves.append(curve_w)
            logger.info("Built synthetic curve for year %d with %d points", year, len(curve_w))
        except Exception as exc:
            logger.warning("Could not parse RLP file for year %d (%s): %s", year, file_path, exc)

    if not yearly_curves:
        raise ValueError("No yearly RLP curves could be parsed")

    merged = pd.concat(yearly_curves, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["ts"], errors="coerce")
    merged = merged.dropna(subset=["ts", "power_w"]).sort_values("ts")
    merged["ts"] = merged["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    logger.info(
        "%s Synthetic consumption curve saved to %s with %d rows %s",
        DASH,
        output_path,
        len(merged),
        DASH,
    )
    return str(output_path)
