"""Seção 23 do v5: diagnóstico dos erros no teste.

Diferença em relação ao v5: o modelo do diagnóstico principal (`main`) é um parâmetro.
No v5 ele estava fixo em "CatBoost_MAE" mesmo quando selecao_modelo.json escolhia outro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .painel import TARGET

ALL_DIAG = ["Naive", "MediaMovel4", "HistGB_MAE", "HistGB_Poisson", "LightGBM_MAE", "LightGBM_MAE_Exog",
            "CatBoost_MAE", "CatBoost_MAE_Exog", "CatBoost_MAE_raw", "CatBoost_Tweedie"]
FOCUS = ["LightGBM_MAE", "LightGBM_MAE_Exog", "CatBoost_MAE", "CatBoost_MAE_Exog", "CatBoost_MAE_raw",
         "Naive", "CatBoost_Tweedie", "MediaMovel4"]


def limiares(hist_train):
    """Pico = P90 e sinal = P10 da receita semanal positiva no treino (nada do teste)."""
    pos = hist_train.loc[hist_train["revenue"] > 0, "revenue"]
    peak = float(pos.quantile(0.90))
    signal = float(pos.quantile(0.10))
    band_edges = [0.0, float(pos.quantile(0.50)), peak]
    seller_peak = hist_train.loc[hist_train["revenue"] > 0].groupby("seller_id")["revenue"].quantile(0.90)
    return {"PEAK_THRESHOLD": peak, "SIGNAL_THRESHOLD": signal, "BAND_EDGES": band_edges,
            "seller_peak_thr": seller_peak, "n_pos": int(len(pos))}


def base_diagnostico(comparison, profile, test, models, thr):
    diag = comparison.merge(profile[["perfil"]], left_on="seller_id", right_index=True, how="left")
    diag = diag.merge(test[["seller_id", "week_start", "revenue_current", "rev_mean_4", "zero_weeks_prev_4"]],
                      on=["seller_id", "week_start"], how="left", validate="one_to_one")
    diag["y"] = diag[TARGET].astype(float)
    diag["pico"] = diag["y"] > thr["PEAK_THRESHOLD"]
    diag["pico_relativo"] = diag["y"] > diag["seller_id"].map(thr["seller_peak_thr"])
    diag["faixa_real"] = pd.cut(diag["y"], [-np.inf, 0, thr["BAND_EDGES"][1], thr["PEAK_THRESHOLD"], np.inf],
                                labels=["zero", "positiva ≤ P50", "P50–P90", "pico (> P90)"])
    for m in models:
        diag[f"err_{m}"] = diag[f"pred_{m}"] - diag["y"]
        diag[f"abs_{m}"] = diag[f"err_{m}"].abs()
    assert diag["perfil"].notna().all() and len(diag) == len(comparison)
    return diag


def diag_metrics(y, p, signal_thr):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ae = np.abs(e)
    zero, pos = y == 0, y > 0
    return {
        "n": len(y), "MAE": ae.mean(),
        "MAE zero": ae[zero].mean() if zero.any() else np.nan,
        "MAE positiva": ae[pos].mean() if pos.any() else np.nan,
        "Viés R$": e.mean(), "Viés %": 100 * e.sum() / y.sum() if y.sum() > 0 else np.nan,
        "Viés R$ (sem receita)": e[zero].mean() if zero.any() else np.nan,
        "Viés R$ (com receita)": e[pos].mean() if pos.any() else np.nan,
        "P90 erro": np.quantile(ae, 0.90), "P95 erro": np.quantile(ae, 0.95), "P99 erro": np.quantile(ae, 0.99),
        "Erro máximo": ae.max(),
        "% falso sinal": 100 * (p[zero] >= signal_thr).mean() if zero.any() else np.nan,
        "% venda não sinalizada": 100 * (p[pos] < signal_thr).mean() if pos.any() else np.nan,
        "% do erro em semanas sem receita": 100 * ae[zero].sum() / ae.sum(),
        "Real médio": y.mean(), "Previsto médio": p.mean(),
    }


def diagnostico_geral(diag, models, signal_thr):
    return pd.DataFrame([{"Modelo": m, **diag_metrics(diag["y"], diag[f"pred_{m}"], signal_thr)} for m in models])


def group_table(df, group_col, models):
    rows = []
    tot_abs = {m: df[f"abs_{m}"].sum() for m in models}
    for g, gg in df.groupby(group_col, observed=True, sort=True):
        for m in models:
            ae, e = gg[f"abs_{m}"], gg[f"err_{m}"]
            rows.append({group_col: g, "modelo": m, "n_linhas": len(gg), "% linhas": 100 * len(gg) / len(df),
                         "real médio": gg["y"].mean(), "previsto médio": gg[f"pred_{m}"].mean(),
                         "MAE": ae.mean(), "viés R$": e.mean(),
                         "viés %": 100 * e.sum() / gg["y"].sum() if gg["y"].sum() > 0 else np.nan,
                         "% do erro absoluto total": 100 * ae.sum() / tot_abs[m]})
    return pd.DataFrame(rows)


def por_vendedor(diag, main, models, scale_final):
    rows = []
    for sid, g in diag.groupby("seller_id", sort=False):
        y = g["y"]
        r = {"seller_id": sid, "perfil": g["perfil"].iloc[0], "semanas": len(g),
             "semanas_com_receita": int((y > 0).sum()), "real_total": y.sum(), "real_medio": y.mean()}
        for m in models:
            r[f"MAE_{m}"] = g[f"abs_{m}"].mean()
            r[f"vies_{m}"] = g[f"err_{m}"].mean()
            r[f"previsto_total_{m}"] = g[f"pred_{m}"].sum()
        r[f"vies_pct_{main}"] = (100 * (r[f"previsto_total_{main}"] - r["real_total"]) / r["real_total"]
                                 if r["real_total"] > 0 else np.nan)
        r[f"erro_abs_total_{main}"] = g[f"abs_{main}"].sum()
        r["skill_vs_naive"] = 1 - r[f"MAE_{main}"] / r["MAE_Naive"] if r["MAE_Naive"] > 0 else np.nan
        r[f"MASE_{main}"] = r[f"MAE_{main}"] / scale_final.get(sid, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def por_perfil(diag, models, signal_thr):
    rows = []
    for perfil in ["intermitente", "medio", "regular"]:
        g = diag.loc[diag["perfil"] == perfil]
        pk = g["pico"]
        for m in models:
            d = diag_metrics(g["y"], g[f"pred_{m}"], signal_thr)
            rows.append({"perfil": perfil, "modelo": m, "n_vendedores": g["seller_id"].nunique(),
                         "MAE": d["MAE"], "MAE zero": d["MAE zero"], "MAE positiva": d["MAE positiva"],
                         "Viés R$": d["Viés R$"], "Viés %": d["Viés %"], "P90 erro": d["P90 erro"],
                         "P95 erro": d["P95 erro"], "n_picos": int(pk.sum()),
                         "% picos subestimados": 100 * (g.loc[pk, f"err_{m}"] < 0).mean() if pk.any() else np.nan,
                         "skill vs Naive": 1 - g[f"abs_{m}"].mean() / g["abs_Naive"].mean()})
    out = pd.DataFrame(rows)
    out["perfil"] = pd.Categorical(out["perfil"], ["intermitente", "medio", "regular"], ordered=True)
    out = out.sort_values(["perfil", "modelo"]).reset_index(drop=True)
    zero_ref = diag.groupby("perfil", observed=True)["y"].mean().rename("MAE de prever zero")
    zero_ref.loc["todos"] = diag["y"].mean()
    return out, zero_ref


def por_semana(diag, models):
    rows = []
    for w, g in diag.groupby("target_week"):
        r = {"target_week": w, "real_total": g["y"].sum(), "vendedores_com_receita": int((g["y"] > 0).sum()),
             "picos": int(g["pico"].sum())}
        for m in models:
            r[f"MAE_{m}"] = g[f"abs_{m}"].mean()
            r[f"vies_{m}"] = g[f"err_{m}"].mean()
            r[f"previsto_total_{m}"] = g[f"pred_{m}"].sum()
        rows.append(r)
    return pd.DataFrame(rows)


def concentracao(diag, models):
    rows = []
    for m in models:
        ae = np.sort(diag[f"abs_{m}"].to_numpy())[::-1]
        rows.append({"modelo": m,
                     "% do erro no 1% maiores": 100 * ae[:int(np.ceil(0.01 * len(ae)))].sum() / ae.sum(),
                     "% do erro nos 5% maiores": 100 * ae[:int(np.ceil(0.05 * len(ae)))].sum() / ae.sum(),
                     "% dos 1% maiores que são subestimação": 100 * (
                         diag.nlargest(int(np.ceil(0.01 * len(diag))), f"abs_{m}")[f"err_{m}"] < 0).mean()})
    return pd.DataFrame(rows)


def peak_table(diag, mask, label, models, peak_thr, seller_peak_thr):
    rows = []
    for m in models:
        g = diag.loc[mask]
        y, p, e = g["y"].to_numpy(), g[f"pred_{m}"].to_numpy(), g[f"err_{m}"].to_numpy()
        thr = peak_thr if label == "global" else diag["seller_id"].map(seller_peak_thr)
        pred_peak = diag[f"pred_{m}"] > thr
        rows.append({
            "criterio": label, "modelo": m, "n_picos": int(mask.sum()),
            "MAE nos picos": np.abs(e).mean(), "Viés nos picos": e.mean(),
            "% picos subestimados": 100 * (e < 0).mean(), "% picos superestimados": 100 * (e > 0).mean(),
            "Real médio nos picos": y.mean(), "Previsto médio nos picos": p.mean(),
            "Previsto/real": p.mean() / y.mean(), "Mediana previsto/real (individual)": np.median(p / y),
            "% picos com previsão acima do limiar (cobertura)": 100 * pred_peak[mask].mean(),
            "n previsões acima do limiar": int(pred_peak.sum()),
            "% previsões acima do limiar que eram pico (precisão)": 100 * mask[pred_peak].mean() if pred_peak.any() else np.nan,
        })
    return pd.DataFrame(rows)
