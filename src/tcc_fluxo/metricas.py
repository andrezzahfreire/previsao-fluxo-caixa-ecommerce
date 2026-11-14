"""Seção 12 do v5: métricas pontuais, escala do MASE e avaliação por vendedor/agregado."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .painel import TARGET


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, p):
    return float(np.mean(np.abs(_arr(y) - _arr(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((_arr(y) - _arr(p)) ** 2)))


def wape(y, p):
    den = np.abs(_arr(y)).sum()
    return float(np.abs(_arr(y) - _arr(p)).sum() / den) if den > 0 else np.nan


def smape(y, p):
    y, p = _arr(y), _arr(p)
    num, den = 2 * np.abs(y - p), np.abs(y) + np.abs(p)
    term = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(term.mean() * 100)


def skill(mae_model, mae_ref):
    return float(1 - mae_model / mae_ref) if mae_ref > 0 else np.nan


def seller_scales(hist_df, cutoff):
    """Escala do MASE (s=1) por vendedor, usando somente semanas <= cutoff. Escala 0 vira NaN."""
    h = hist_df.loc[hist_df["week_start"] <= cutoff, ["seller_id", "week_start", "revenue"]]
    d = h.groupby("seller_id")["revenue"].diff().abs()
    sc = d.groupby(h["seller_id"]).mean()
    return sc.where(sc > 0)


def aggregate_scale(hist_df, cutoff):
    tot = hist_df.loc[hist_df["week_start"] <= cutoff].groupby("week_start")["revenue"].sum()
    return float(tot.diff().abs().mean())


def evaluate_seller_level(df, models, scale, ref="Naive"):
    """Retorna (métricas agrupadas vendedor-semana, métricas por vendedor)."""
    y = df[TARGET].to_numpy(float)
    sc = df["seller_id"].map(scale).to_numpy(float)
    ok = np.isfinite(sc)
    e_ref = np.abs(y - df[f"pred_{ref}"].to_numpy(float))
    pooled, per_seller = [], []
    for m in models:
        p = df[f"pred_{m}"].to_numpy(float)
        e = np.abs(y - p)
        pooled.append({"modelo": m, "MAE": e.mean(), "RMSE": rmse(y, p), "WAPE": wape(y, p),
                       "sMAPE": smape(y, p), "MASE": float(np.mean(e[ok] / sc[ok])),
                       "skill_vs_naive": skill(e.mean(), e_ref.mean())})
        den = np.abs(y) + np.abs(p)
        tmp = pd.DataFrame({"seller_id": df["seller_id"].to_numpy(), "ae": e, "se": (y - p) ** 2, "ay": np.abs(y),
                            "sm": np.divide(2 * e, den, out=np.zeros_like(e), where=den > 0),
                            "ae_ref": e_ref, "scale": sc})
        s = tmp.groupby("seller_id").agg(n=("ae", "size"), sae=("ae", "sum"), sse=("se", "sum"), sy=("ay", "sum"),
                                         ssm=("sm", "sum"), sae_ref=("ae_ref", "sum"), scale=("scale", "first"))
        s["MAE"] = s["sae"] / s["n"]
        s["RMSE"] = np.sqrt(s["sse"] / s["n"])
        s["WAPE"] = s["sae"] / s["sy"].where(s["sy"] > 0)
        s["sMAPE"] = 100 * s["ssm"] / s["n"]
        s["MASE"] = s["MAE"] / s["scale"]
        s["skill_vs_naive"] = 1 - s["sae"] / s["sae_ref"].where(s["sae_ref"] > 0)
        s["modelo"] = m
        per_seller.append(s.reset_index())
    return pd.DataFrame(pooled), pd.concat(per_seller, ignore_index=True)


def summarize_sellers(per_seller):
    """Distribuição das métricas entre vendedores (nível primário)."""
    rows = []
    for m, g in per_seller.groupby("modelo", sort=False):
        sk = g["skill_vs_naive"].dropna()
        row = {"modelo": m, "n_vendedores": g["seller_id"].nunique()}
        for met in ("MASE", "WAPE", "MAE", "skill_vs_naive"):
            v = g[met].replace([np.inf, -np.inf], np.nan).dropna()
            row.update({f"{met}_media": v.mean(), f"{met}_mediana": v.median(), f"{met}_P25": v.quantile(0.25),
                        f"{met}_P75": v.quantile(0.75), f"{met}_P90": v.quantile(0.90)})
        row["prop_vendedores_melhora"] = float((sk > 0).mean()) if len(sk) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_aggregate(df, models, agg_scale, ref="Naive"):
    a = df.groupby("target_week")[[TARGET] + [f"pred_{m}" for m in models]].sum()
    y = a[TARGET].to_numpy(float)
    base = mae(y, a[f"pred_{ref}"])
    rows = []
    for m in models:
        p = a[f"pred_{m}"].to_numpy(float)
        rows.append({"modelo": m, "MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p), "sMAPE": smape(y, p),
                     "MASE": mae(y, p) / agg_scale, "skill_vs_naive": skill(mae(y, p), base)})
    return pd.DataFrame(rows), a
