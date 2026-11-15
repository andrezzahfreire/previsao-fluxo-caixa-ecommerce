"""Notebook 13: números e figuras pedidos pelo texto que não saíam de nenhum notebook.

Não altera nenhuma escolha: lê artefatos já gravados (coorte, painel, validação, previsões do teste).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inferencia import block_bootstrap_mean, dm_hln
from .metricas import evaluate_seller_level, seller_scales
from .painel import TARGET


# ------------------------------------------------------------------ 1. funil da coorte
def funil(elig, weekly_observed, train_end, min_calendar_weeks, min_sales_weeks):
    """Etapas da coorte. Os dois critérios são aplicados em sequência (calendário, depois vendas)."""
    n_total = len(elig)
    ok_cal = elig["ok_calendar"]
    ok_ambos = ok_cal & elig["ok_sales"]
    so_2018 = set(weekly_observed.loc[weekly_observed["week_start"] > train_end, "seller_id"]) - set(elig.index)
    etapas = pd.DataFrame([
        {"etapa": "Vendedores com venda no treino (2017)", "vendedores": n_total, "excluidos_nesta_etapa": 0},
        {"etapa": f"≥ {min_calendar_weeks} semanas de calendário", "vendedores": int(ok_cal.sum()),
         "excluidos_nesta_etapa": int((~ok_cal).sum())},
        {"etapa": f"≥ {min_sales_weeks} semanas com venda (coorte)", "vendedores": int(ok_ambos.sum()),
         "excluidos_nesta_etapa": int((ok_cal & ~elig["ok_sales"]).sum())},
    ])
    extra = {"vendedores_so_em_2018 (fora por construção)": len(so_2018),
             "falham_so_calendario": int((~ok_cal & elig["ok_sales"]).sum()),
             "falham_so_vendas": int((ok_cal & ~elig["ok_sales"]).sum()),
             "falham_os_dois": int((~ok_cal & ~elig["ok_sales"]).sum())}
    return etapas, extra


# ------------------------------------------------------------------ 2. Tabela W (perfis)
def tabela_perfis(serie, perfis, train_target_end):
    """Por perfil: frequência de venda (definição do perfil), receita semanal mediana e % de zeros."""
    per = perfis.set_index("seller_id")
    s = serie.assign(perfil=serie["seller_id"].map(per["perfil"]).astype(str))
    tr = s.loc[s["week_start"] <= train_target_end]
    te = s.loc[s["week_start"] > train_target_end]
    rows = []
    for p in ["intermitente", "medio", "regular", "todos"]:
        a = tr if p == "todos" else tr.loc[tr["perfil"] == p]
        b = te if p == "todos" else te.loc[te["perfil"] == p]
        f = per["freq_venda_treino"] if p == "todos" else per.loc[per["perfil"] == p, "freq_venda_treino"]
        rows.append({"perfil": p, "vendedores": int(a["seller_id"].nunique()),
                     "faixa_freq_venda_treino": f"{f.min():.2f} a {f.max():.2f}",
                     "% semanas com venda no treino (média da freq_venda_treino)": 100 * f.mean(),
                     "receita semanal mediana no treino, semanas com venda (R$)": a.loc[a["revenue"] > 0, "revenue"].median(),
                     "receita semanal mediana no treino, todas as semanas (R$)": a["revenue"].median(),
                     "% semanas com zero no treino": 100 * (a["revenue"] == 0).mean(),
                     "% semanas com zero no teste": 100 * (b["revenue"] == 0).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. validação × teste
def validacao_vs_teste(cv_detalhe, resumo_val, comparison, hist, origens, train_target_end, modelos):
    """MASE médio por vendedor do Naive e do escolhido em cada origem, na média da validação e no teste,
    com a escala mediana do MASE e o tamanho mediano do histórico em cada corte."""
    v = cv_detalhe.loc[cv_detalhe["modelo"].isin(modelos)].copy()
    v["origem"] = pd.to_datetime(v["origem"])
    mae_naive = v.loc[v["modelo"] == "Naive"].set_index("origem")["MAE"]
    v = v.pivot_table(index="origem", columns="modelo", values="MASE_vend_media")
    rows = []
    for o in origens:
        o = pd.Timestamp(o)
        h = hist.loc[hist["week_start"] <= o]
        sc = seller_scales(hist, o).dropna()
        rows.append({"periodo": f"validação, origem {o:%d/%m/%Y}", **{m: float(v.loc[o, m]) for m in modelos},
                     "MAE_Naive_R$": float(mae_naive.loc[o]), "media_1_sobre_escala": float((1 / sc).mean()),
                     "escala_mediana_R$": float(sc.median()),
                     "semanas_mediana_no_historico": float(h.groupby("seller_id").size().median())})
    rv = resumo_val.set_index("modelo")
    rows.append({"periodo": "validação, média das origens", **{m: float(rv.loc[m, "MASE_vend_media"]) for m in modelos},
                 "MAE_Naive_R$": float(rv.loc["Naive", "MAE_media"])})
    scale = seller_scales(hist, train_target_end)
    pooled, per = evaluate_seller_level(comparison, modelos, scale)
    rows.append({"periodo": "teste (escala com todo o treino)",
                 "MAE_Naive_R$": float(pooled.set_index("modelo").loc["Naive", "MAE"]),
                 "media_1_sobre_escala": float((1 / scale.dropna()).mean()),
                 **{m: float(per.loc[per["modelo"] == m, "MASE"].replace([np.inf, -np.inf], np.nan).mean())
                    for m in modelos},
                 "escala_mediana_R$": float(scale.median()),
                 "semanas_mediana_no_historico": float(hist.groupby("seller_id").size().median())})
    out = pd.DataFrame(rows)
    out["razao_escolhido_sobre_Naive"] = out[modelos[1]] / out[modelos[0]]
    return out


# ------------------------------------------------------------------ 4. ablação das exógenas
def ablacao(comparison, hist, train_target_end, sem, com, n_boot, seed):
    """Modelo sem exógenas × com exógenas no teste: MASE por vendedor (bootstrap pareado por vendedor)
    e Diebold-Mariano HLN nas perdas semanais (média entre vendedores e erro do total)."""
    rng = np.random.default_rng(seed)
    scale = seller_scales(hist, train_target_end)
    _, per = evaluate_seller_level(comparison, [sem, com], scale)
    w = per.pivot(index="seller_id", columns="modelo", values="MASE").dropna()
    d_vend = (w[com] - w[sem]).to_numpy()
    idx = rng.integers(0, len(d_vend), size=(n_boot, len(d_vend)))
    boots = d_vend[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    resumo = [{"nivel": "MASE médio por vendedor", sem: w[sem].mean(), com: w[com].mean(),
               "diferenca (com - sem)": d_vend.mean(), "ic95_inf": lo, "ic95_sup": hi,
               "teste": "bootstrap pareado por vendedor", "estatistica": np.nan,
               "p_valor": float(np.mean(np.abs(boots - d_vend.mean()) >= abs(d_vend.mean()))),
               "% vendedores em que com < sem": 100 * float((d_vend < 0).mean())}]

    e = pd.DataFrame({"w": comparison["target_week"],
                      sem: (comparison[TARGET] - comparison[f"pred_{sem}"]).abs(),
                      com: (comparison[TARGET] - comparison[f"pred_{com}"]).abs()})
    Ls = e.groupby("w")[[sem, com]].mean()
    agg = comparison.groupby("target_week")[[TARGET, f"pred_{sem}", f"pred_{com}"]].sum()
    La = pd.DataFrame({m: (agg[TARGET] - agg[f"pred_{m}"]).abs() for m in (sem, com)})
    for nivel, L, ref in [("perda absoluta média por vendedor-semana (R$)", Ls, None),
                          ("erro absoluto do total semanal (R$)", La, agg[TARGET])]:
        d = (L[com] - L[sem]).to_numpy()
        stat, p = dm_hln(d)
        _, lo, hi, _ = block_bootstrap_mean(d, n_boot, rng)
        linha = {"nivel": nivel, sem: L[sem].mean(), com: L[com].mean(), "diferenca (com - sem)": d.mean(),
                 "ic95_inf": lo, "ic95_sup": hi, "teste": f"Diebold-Mariano HLN, {len(d)} semanas",
                 "estatistica": stat, "p_valor": p, "% vendedores em que com < sem": np.nan}
        rows_wape = None
        if ref is not None:
            rows_wape = {"nivel": "WAPE agregado", sem: L[sem].sum() / ref.sum(), com: L[com].sum() / ref.sum(),
                         "diferenca (com - sem)": (L[com].sum() - L[sem].sum()) / ref.sum(), "teste": "descritivo"}
        resumo.append(linha)
        if rows_wape:
            resumo.append(rows_wape)
    return pd.DataFrame(resumo)
