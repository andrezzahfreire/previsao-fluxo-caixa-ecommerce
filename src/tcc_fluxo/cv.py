"""Seções 14–16 do v5: validação com origens móveis (só treino), estabilidade do Ridge e seleção."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .metricas import evaluate_seller_level, seller_scales, summarize_sellers
from .modelos import check_and_clip, fit_predict

QUANTS = [0, 0.5, 0.9, 0.99, 0.999, 1]


def origens_cv(C):
    all_origins = pd.date_range(C.CV_FIRST_ORIGIN, C.TRAIN_TARGET_END - pd.Timedelta(weeks=1),
                                freq=f"{7 * C.CV_STEP_WEEKS}D")
    cv_origins = pd.DatetimeIndex([o for o in all_origins
                                   if o + pd.Timedelta(weeks=C.CV_VAL_WEEKS) <= C.TRAIN_TARGET_END])
    dropped = [o.date() for o in all_origins if o not in cv_origins]
    assert all(o.weekday() == 0 for o in cv_origins)
    return cv_origins, dropped


def rodar_cv(train, hist, cv_origins, models, spec, C, verbose=True):
    """Para cada origem O: ajusta com target_week <= O e valida nas CV_VAL_WEEKS semanas seguintes."""
    T = spec.target
    cv_rows, ridge_q_rows = [], []
    t0 = time.time()
    for origin in cv_origins:
        val_end = origin + pd.Timedelta(weeks=C.CV_VAL_WEEKS)
        fit = train.loc[train["target_week"] <= origin]
        val = train.loc[(train["target_week"] > origin) & (train["target_week"] <= val_end)].copy()
        assert val["target_week"].max() <= C.TRAIN_TARGET_END
        assert val["target_week"].nunique() == C.CV_VAL_WEEKS, "origem com validação incompleta"
        scale_o = seller_scales(hist, origin)
        max_fit_target = float(fit[T].max())
        infos = {}
        for name in models:
            pred, info = check_and_clip(fit_predict(name, fit, val, spec))
            val[f"pred_{name}"] = pred
            info["pred_max"] = float(np.nanmax(np.where(np.isfinite(pred), pred, np.nan)))
            infos[name] = info
            if name.startswith("Ridge_a"):
                q = np.quantile(pred[np.isfinite(pred)], QUANTS)
                ridge_q_rows.append({"origem": origin.date(), "modelo": name,
                                     **{f"q{qq}": v for qq, v in zip(QUANTS, q)},
                                     "max_alvo_treino": max_fit_target, **info})
        pooled, per_seller = evaluate_seller_level(val, models, scale_o)
        summ = summarize_sellers(per_seller).set_index("modelo")
        for _, r in pooled.iterrows():
            m = r["modelo"]
            cv_rows.append({
                "origem": origin.date(), "inicio_val": (origin + pd.Timedelta(weeks=1)).date(),
                "fim_val": val_end.date(), "semanas_val": val["target_week"].nunique(),
                "linhas_treino": len(fit), "linhas_val": len(val), "modelo": m,
                "MAE": r["MAE"], "RMSE": r["RMSE"], "WAPE": r["WAPE"], "sMAPE": r["sMAPE"],
                "MASE_agrupado": r["MASE"], "skill_vs_naive": r["skill_vs_naive"],
                "MASE_vend_media": summ.loc[m, "MASE_media"], "MASE_vend_mediana": summ.loc[m, "MASE_mediana"],
                "WAPE_vend_media": summ.loc[m, "WAPE_media"],
                "skill_vend_mediana": summ.loc[m, "skill_vs_naive_mediana"],
                "prop_vend_melhora": summ.loc[m, "prop_vendedores_melhora"],
                "n_vendedores": summ.loc[m, "n_vendedores"],
                "pred_max": infos[m]["pred_max"], "pred_nao_finitas": infos[m]["n_nao_finitas"],
                "pred_negativas_brutas": infos[m]["n_negativas_brutas"], "max_alvo_treino": max_fit_target,
            })
        if verbose:
            print(f"origem {origin.date()} | treino {len(fit):>6,} | val {len(val):>5,} "
                  f"({val['target_week'].nunique()} sem.) | {time.time() - t0:5.0f}s")
    return pd.DataFrame(cv_rows), pd.DataFrame(ridge_q_rows)


def estabilidade_ridge(cv_detail, ridge_names):
    """Seção 15: alpha instável se gerar previsão não finita ou acima do maior alvo do treino da origem."""
    stab = (cv_detail.loc[cv_detail["modelo"].isin(ridge_names)]
            .assign(acima_max_treino=lambda d: d["pred_max"] > d["max_alvo_treino"])
            .groupby("modelo", sort=False)
            .agg(MASE_vend_media=("MASE_vend_media", "mean"), WAPE_vend_media=("WAPE_vend_media", "mean"),
                 skill_medio=("skill_vs_naive", "mean"), pred_max=("pred_max", "max"),
                 max_alvo_treino=("max_alvo_treino", "max"),
                 origens_acima_max=("acima_max_treino", "sum"), nao_finitas=("pred_nao_finitas", "sum")))
    stab["estavel"] = (stab["origens_acima_max"] == 0) & (stab["nao_finitas"] == 0)
    stable = stab.loc[stab["estavel"]]
    if len(stable):
        return stab, stable["MASE_vend_media"].idxmin(), True
    return stab, stab["MASE_vend_media"].idxmin(), False


def resumo_validacao(cv_detail, boost_models, ridge_name, ridge_eligible):
    r = (cv_detail.groupby("modelo", sort=False)
         .agg(MAE_media=("MAE", "mean"), MAE_mediana=("MAE", "median"), MAE_dp=("MAE", "std"),
              RMSE_media=("RMSE", "mean"), WAPE_media=("WAPE", "mean"), WAPE_dp=("WAPE", "std"),
              sMAPE_media=("sMAPE", "mean"),
              MASE_vend_media=("MASE_vend_media", "mean"), MASE_vend_media_dp=("MASE_vend_media", "std"),
              MASE_vend_mediana=("MASE_vend_mediana", "mean"), WAPE_vend_media=("WAPE_vend_media", "mean"),
              skill_media=("skill_vs_naive", "mean"), skill_dp=("skill_vs_naive", "std"),
              prop_vend_melhora=("prop_vend_melhora", "mean"), n_origens=("origem", "nunique"))
         .reset_index())
    r["familia"] = r["modelo"].str.replace(r"_a\d+$", "", regex=True)
    r["candidato"] = r["modelo"].isin(["Naive", "MediaMovel4", *boost_models] + ([ridge_name] if ridge_eligible else []))
    return r


def selecionar(resumo, cv_origins, C, ridge_name, ridge_eligible, painel_sha256):
    """Seção 16: menor MASE médio por vendedor; empate prático -> modelo mais simples."""
    cand = resumo.loc[resumo["candidato"]].copy()
    metric = "MASE_vend_media" if cand["MASE_vend_media"].notna().all() else "WAPE_vend_media"
    best = cand[metric].min()
    tied = cand.loc[cand[metric] <= best * (1 + C.PRACTICAL_TIE_TOL)].copy()
    tied["complexidade"] = tied["familia"].map(C.MODEL_COMPLEXITY)
    chosen = tied.sort_values(["complexidade", metric]).iloc[0]["modelo"]
    return {
        "modelo_escolhido": chosen,
        "criterio": f"menor {metric} (média entre {len(cv_origins)} origens), "
                    f"empate prático <= {C.PRACTICAL_TIE_TOL:.0%} -> mais simples",
        "valor_criterio": float(cand.set_index("modelo").loc[chosen, metric]),
        "melhor_valor_criterio": float(best),
        "modelos_em_empate_pratico": tied["modelo"].tolist(),
        "candidatos": cand["modelo"].tolist(),
        "ridge_elegivel": bool(ridge_eligible), "ridge_alpha_documentado": ridge_name,
        "origens": [str(o.date()) for o in cv_origins],
        "registrado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "teste_usado_na_selecao": False,
        "painel_sha256": painel_sha256,
    }
