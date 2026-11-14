"""Seção 5.1 do v5: macroeconomia (SGS/BCB) e calendário conhecido ex ante."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _parse_sgs_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(".", "").replace(",", ".")
    try:
        direct = float(str(value).strip())
        if "." in str(value) and "," not in str(value):
            return direct
    except Exception:
        pass
    try:
        return float(text)
    except Exception:
        return np.nan


def fetch_sgs_monthly(variable, code, start, end, cache_dir: Path):
    """Lê do cache; se não houver, baixa da API do SGS e grava o cache.

    O cache fica em dados_brutos/macro_cache (fora da pasta da execução): assim todas as
    execuções usam a MESMA vintage das séries, que o BCB revisa ao longo do tempo.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sgs_{code}_{variable}.csv"
    if cache_path.is_file():
        return pd.read_csv(cache_path, parse_dates=["date", "month"])
    import requests
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    params = {"formato": "json", "dataInicial": start.strftime("%d/%m/%Y"), "dataFinal": end.strftime("%d/%m/%Y")}
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"SGS não retornou dados para {variable} (série {code}).")
    out = pd.DataFrame(payload).rename(columns={"data": "date", "valor": "value"})
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out["value"] = out["value"].map(_parse_sgs_value)
    out = out.dropna(subset=["date", "value"]).copy()
    out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
    out["variavel"] = variable
    out["codigo_sgs"] = int(code)
    out = out[["date", "month", "variavel", "codigo_sgs", "value"]].sort_values("date")
    out.to_csv(cache_path, index=False)
    return out


def montar_macro(macro_series: dict, start, end, cache_dir):
    parts = [fetch_sgs_monthly(v, c, start, end, cache_dir) for v, c in macro_series.items()]
    macro_long = pd.concat(parts, ignore_index=True)
    macro_wide = macro_long.pivot_table(index="month", columns="variavel", values="value", aggfunc="last").sort_index()
    expected = pd.date_range(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp(), freq="MS")
    missing = sorted(set(expected) - set(macro_wide.index))
    if missing:
        raise RuntimeError(f"Meses macroeconômicos ausentes na API/cache: {missing}")
    if macro_wide[list(macro_series)].isna().any().any():
        raise RuntimeError("Há valores macroeconômicos ausentes após a consolidação das séries.")
    return macro_long, macro_wide


def easter_sunday(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def nth_weekday(year, month, weekday, n):
    first = pd.Timestamp(year=year, month=month, day=1)
    delta = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=delta + 7 * (n - 1))


def fourth_friday_november(year):
    return nth_weekday(year, 11, weekday=4, n=4)


def calendario_eventos(years):
    rows = []
    for year in years:
        easter = easter_sunday(year)
        dates = {
            "holiday_new_year": pd.Timestamp(year=year, month=1, day=1),
            "holiday_carnival": easter - pd.Timedelta(days=47),
            "holiday_good_friday": easter - pd.Timedelta(days=2),
            "holiday_easter": easter,
            "holiday_corpus_christi": easter + pd.Timedelta(days=60),
            "holiday_tiradentes": pd.Timestamp(year=year, month=4, day=21),
            "holiday_labor_day": pd.Timestamp(year=year, month=5, day=1),
            "commercial_mothers_day": nth_weekday(year, 5, weekday=6, n=2),
            "commercial_valentines_day": pd.Timestamp(year=year, month=6, day=12),
            "commercial_fathers_day": nth_weekday(year, 8, weekday=6, n=2),
            "holiday_independence": pd.Timestamp(year=year, month=9, day=7),
            "holiday_children_day": pd.Timestamp(year=year, month=10, day=12),
            "holiday_all_souls": pd.Timestamp(year=year, month=11, day=2),
            "holiday_republic": pd.Timestamp(year=year, month=11, day=15),
            "commercial_black_friday": fourth_friday_november(year),
            "holiday_christmas": pd.Timestamp(year=year, month=12, day=25),
            "commercial_consumer_day": pd.Timestamp(year=year, month=3, day=15),
        }
        for event, date in dates.items():
            rows.append({"date": date, "evento": event, "ano": year})
    return pd.DataFrame(rows).drop_duplicates(["date", "evento"]).sort_values("date")


def exogenas_semanais(events, macro_wide, macro_series, panel_start, test_end, macro_lag_months):
    """Atributos exógenos por semana-alvo. A semana-alvo t+1 usa a macro do mês de (t) menos o atraso."""
    cw = pd.DataFrame({"target_week": pd.date_range(panel_start, test_end, freq="7D")})
    for event in events["evento"].unique():
        event_weeks = events.loc[events["evento"] == event, "date"].map(lambda d: d - pd.Timedelta(days=d.weekday()))
        cw[f"cal_{event}"] = cw["target_week"].isin(set(event_weeks)).astype("float64")
    holiday = [c for c in cw.columns if c.startswith("cal_holiday_")]
    commercial = [c for c in cw.columns if c.startswith("cal_commercial_")]
    cw["cal_holiday_count"] = cw[holiday].sum(axis=1)
    cw["cal_commercial_count"] = cw[commercial].sum(axis=1)
    cw["cal_any_holiday"] = (cw["cal_holiday_count"] > 0).astype("float64")
    cw["cal_any_commercial"] = (cw["cal_commercial_count"] > 0).astype("float64")
    cw["cal_event_count"] = cw[holiday + commercial].sum(axis=1)
    calendar_features = holiday + commercial + ["cal_holiday_count", "cal_commercial_count",
                                                "cal_any_holiday", "cal_any_commercial", "cal_event_count"]

    cw["reference_week"] = cw["target_week"] - pd.Timedelta(weeks=1)
    cw["macro_reference_month"] = (cw["reference_week"].dt.to_period("M").dt.to_timestamp()
                                   - pd.DateOffset(months=macro_lag_months))
    mw = cw[["target_week", "reference_week", "macro_reference_month"] + calendar_features].merge(
        macro_wide.reset_index(), left_on="macro_reference_month", right_on="month", how="left",
        validate="many_to_one").drop(columns=["month"])
    exog_features = list(macro_series) + calendar_features
    exog = mw[["target_week", "reference_week", "macro_reference_month"] + exog_features].copy()
    assert exog[exog_features].notna().all().all()
    listas = {"EXOGENOUS_FEATURES": exog_features, "CALENDAR_FEATURES": calendar_features,
              "HOLIDAY_FEATURES": holiday, "COMMERCIAL_FEATURES": commercial,
              "MACRO_FEATURES": list(macro_series)}
    return exog, listas
