"""Seções 17, 20, 22 e 25 do v5: avaliação única no teste, perfis, sensibilidade e tabela final."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .metricas import evaluate_aggregate, evaluate_seller_level, summarize_sellers
from .modelos import check_and_clip, fit_predict
from .painel import TARGET

QUANTS = [0, 0.5, 0.9, 0.99, 0.999, 1]


def prever_teste(train, test, final_models, eligible_final, spec, exog_features):
    """Reestima cada modelo com todo o treino e prevê o teste uma única vez."""
    test = test.copy()
    checks = []
    t0 = time.time()
    for name in final_models:
        pred, info = check_and_clip(fit_predict(name, train, test, spec))
        test[f"pred_{name}"] = pred
        checks.append({"modelo": name, **info, "pred_min": np.nanmin(pred), "pred_max": np.nanmax(pred)})
    print(f"Modelos ajustados no treino completo em {time.time() - t0:.0f}s")
    for m in eligible_final:
        assert np.isfinite(test[f"pred_{m}"]).all() and (test[f"pred_{m}"] >= 0).all(), m
    comparison = test[["seller_id", "week_start", "target_week", TARGET] + exog_features].copy()
    for m in final_models:
        comparison[f"pred_{m}"] = test[f"pred_{m}"].to_numpy()
    return test, comparison, pd.DataFrame(checks)


def auditoria_determinismo(train, test, boost_models, spec, tol):
    rows = []
    for name in ["LightGBM_MAE", "LightGBM_MAE_Exog", "CatBoost_MAE", "CatBoost_MAE_Exog", "CatBoost_MAE_raw"]:
        if name not in boost_models:
            continue
        p1, _ = check_and_clip(fit_predict(name, train, test, spec))
        p2, _ = check_and_clip(fit_predict(name, train, test, spec))
        delta = np.abs(p1 - p2)
        mx = float(np.nanmax(delta))
        rows.append({"modelo": name, "thread_count": spec.n_threads, "seed": spec.seed, "max_abs_diff": mx,
                     "n_diferentes": int((delta > tol).sum()), "tolerancia": tol, "deterministico": bool(mx <= tol)})
    return pd.DataFrame(rows)


def metricas_teste(comparison, final_models, eligible_final, chosen, scale_final, agg_scale_final):
    pooled, per_seller = evaluate_seller_level(comparison, final_models, scale_final)
    seller_summary = summarize_sellers(per_seller)
    flags = pd.DataFrame({"modelo": final_models})
    flags["elegivel_selecao"] = flags["modelo"].isin(eligible_final)
    flags["escolhido_validacao"] = flags["modelo"].eq(chosen)
    mvs = flags.merge(pooled, on="modelo").merge(
        seller_summary[["modelo", "MASE_media", "MASE_mediana", "WAPE_media", "skill_vs_naive_mediana",
                        "prop_vendedores_melhora"]], on="modelo")
    agg_test, aggregate = evaluate_aggregate(comparison, final_models, agg_scale_final)
    agg_test = flags.merge(agg_test, on="modelo")
    return {"pooled": pooled, "per_seller": per_seller, "seller_summary": seller_summary,
            "vendedor_semana": mvs, "agregado": agg_test, "aggregate": aggregate, "flags": flags}


def vies_agregado(aggregate, models):
    """Acréscimo da v6: viés do total semanal (previsto − real). Mostra a subestimação sistemática
    que uma perda MAE (mediana condicional) produz quando a série é intermitente."""
    y = aggregate[TARGET].to_numpy(float)
    rows = []
    for m in models:
        p = aggregate[f"pred_{m}"].to_numpy(float)
        rows.append({"modelo": m, "real_total": y.sum(), "previsto_total": p.sum(),
                     "vies_pct_total": 100 * (p.sum() - y.sum()) / y.sum(),
                     "semanas_subestimadas": int((p < y).sum()), "semanas": len(y)})
    return pd.DataFrame(rows)


def metricas_por_perfil(per_seller_test, profile, chosen, eligible_final):
    ps = per_seller_test.merge(profile[["perfil"]], left_on="seller_id", right_index=True, how="left")
    assert ps["perfil"].notna().all()
    rows = []
    for (perfil, m), g in ps.groupby(["perfil", "modelo"], observed=True, sort=False):
        sk = g["skill_vs_naive"].dropna()
        rows.append({"perfil": perfil, "modelo": m, "n_vendedores": g["seller_id"].nunique(),
                     "n_linhas": int(g["n"].sum()), "MAE": g["sae"].sum() / g["n"].sum(),
                     "WAPE": g["sae"].sum() / g["sy"].sum(),
                     "MASE_medio": g["MASE"].replace([np.inf, -np.inf], np.nan).mean(),
                     "skill_vs_naive": 1 - g["sae"].sum() / g["sae_ref"].sum(),
                     "mediana_ganho_vendedor": sk.median(), "prop_vendedores_melhora": (sk > 0).mean(),
                     "escolhido_validacao": m == chosen})
    df = pd.DataFrame(rows)
    df["perfil"] = pd.Categorical(df["perfil"], ["intermitente", "medio", "regular"], ordered=True)
    return df.sort_values(["perfil", "modelo"]).reset_index(drop=True)


def sensibilidade(comparison, eligible_final, test_end, scale_final, agg_scale_final,
                  order_seller_revenue, cohort_index):
    cmp_s = comparison.loc[comparison["target_week"] < test_end]
    pooled_s, per_seller_s = evaluate_seller_level(cmp_s, eligible_final, scale_final)
    summ_s = summarize_sellers(per_seller_s)
    agg_s, _ = evaluate_aggregate(cmp_s, eligible_final, agg_scale_final)
    sens = (pooled_s[["modelo", "MAE", "WAPE", "skill_vs_naive"]]
            .merge(summ_s[["modelo", "MASE_media", "prop_vendedores_melhora"]], on="modelo")
            .merge(agg_s[["modelo", "WAPE", "skill_vs_naive"]].rename(
                columns={"WAPE": "WAPE_agregado", "skill_vs_naive": "skill_agregado"}), on="modelo"))
    test_weeks = set(comparison["target_week"])
    osr = order_seller_revenue.loc[order_seller_revenue["week_start"].isin(test_weeks)
                                   & order_seller_revenue["seller_id"].isin(cohort_index)]
    share_nd = osr.loc[osr["order_status"].ne("delivered"), "allocated_revenue"].sum() / osr["allocated_revenue"].sum()
    return sens, float(share_nd), int(cmp_s["target_week"].nunique())


def tabela_final(final_models, eligible_final, chosen, resumo_val, ci, seller_summary, agg_test):
    val = resumo_val.set_index("modelo")

    def _ci_txt(m, met, nivel_prefix, pct=True):
        r = ci.loc[(ci["modelo"] == m) & (ci["metrica"] == met) & ci["nivel"].str.startswith(nivel_prefix)].iloc[0]
        f = (lambda v: f"{v:+.1%}") if pct else (lambda v: f"{v:.3f}")
        return f"{f(r['estimativa'])} [{f(r['ic95_inf'])}; {f(r['ic95_sup'])}]"

    ss = seller_summary.set_index("modelo")
    ag = agg_test.set_index("modelo")
    return pd.DataFrame({
        "modelo": final_models,
        "elegível": [m in eligible_final for m in final_models],
        "escolhido": ["◀" if m == chosen else "" for m in final_models],
        "VAL MASE médio vend.": [val.loc[m, "MASE_vend_media"] for m in final_models],
        "TESTE MASE médio vend. [IC95]": [_ci_txt(m, "MASE médio por vendedor", "vendedor", pct=False) for m in final_models],
        "TESTE MASE mediano vend.": [ss.loc[m, "MASE_mediana"] for m in final_models],
        "TESTE skill vend.-semana [IC95]": [_ci_txt(m, "skill vs Naive (agrupado)", "vendedor") for m in final_models],
        "TESTE % vend. que melhoram": [ss.loc[m, "prop_vendedores_melhora"] for m in final_models],
        "TESTE WAPE agregado": [ag.loc[m, "WAPE"] for m in final_models],
        "TESTE skill agregado [IC95]": [_ci_txt(m, "skill vs Naive", "agregado") for m in final_models],
    })
