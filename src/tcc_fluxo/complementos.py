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


# ------------------------------------------------------------------ 6. ARIMA(X) por vendedor: previsões extremas
def diagnostico_estat_extremos(pred, comparison, hist, perfis, mase_ref=None, fatores=(10, 5),
                               modelos=("ARIMA_vendedor", "SARIMA_vendedor", "ARIMAX_vendedor", "SARIMAX_vendedor")):
    """Por que ARIMAX/SARIMAX por vendedor somados têm erro agregado tão alto.

    pred: previsões brutas por vendedor-semana (artefato da 08a, antes do corte em zero).
    Mede (i) vendedores cuja previsão máxima no teste excede `fator` × a maior receita semanal do próprio
    vendedor no treino e quanto do erro vem deles; (ii) o efeito do corte das previsões negativas em zero.
    mase_ref: dict modelo -> MASE agregado da 08c; o MASE sem os extremos é proporcional ao MAE (mesma escala).
    """
    y = "target_next_week"
    d = pred.merge(comparison[["seller_id", "target_week", y]], on=["seller_id", "target_week"], how="left")
    assert d[y].notna().all()
    max_treino = hist.groupby("seller_id")["revenue"].max()
    perfil = perfis.set_index("seller_id")["perfil"].astype(str)
    resumo, casos = [], []

    def mae_agregado(p):
        a = d.assign(p=p).groupby("target_week")[[y, "p"]].sum()
        return float((a["p"] - a[y]).abs().mean()), float(a["p"].sum() / a[y].sum() - 1)

    for m in modelos:
        bruto = d[f"pred_{m}"].to_numpy(float)
        p = np.clip(bruto, 0, None)
        mae, vies = mae_agregado(p)
        mae_sem_corte, vies_sem_corte = mae_agregado(bruto)
        g = (d.assign(p=p, ae=np.abs(p - d[y])).groupby("seller_id")
             .agg(prev_max=("p", "max"), erro_abs_soma=("ae", "sum"), real_soma=(y, "sum"), prev_soma=("p", "sum")))
        g["max_receita_treino"] = g.index.map(max_treino)
        g["razao_max"] = g["prev_max"] / g["max_receita_treino"]
        g["perfil"] = g.index.map(perfil)
        base = {"modelo": m, "MAE_agregado": mae, "vies_total_%": 100 * vies,
                "previsoes_negativas_zeradas": int((bruto < 0).sum()),
                "%_previsoes_negativas": 100 * float((bruto < 0).mean()),
                "MAE_agregado_sem_corte_em_zero": mae_sem_corte, "vies_total_sem_corte_%": 100 * vies_sem_corte}
        for k in fatores:
            X = g.index[g["razao_max"] > k]
            p2 = np.where(d["seller_id"].isin(X), d[y], p)        # erro desses vendedores zerado
            mae2, _ = mae_agregado(p2)
            cont = g.loc[X, "perfil"].value_counts()
            linha = {**base, "fator": k, "vendedores_acima": len(X),
                     "intermitentes": int(cont.get("intermitente", 0)), "medios": int(cont.get("medio", 0)),
                     "regulares": int(cont.get("regular", 0)),
                     "%_do_erro_abs_somado": 100 * float(g.loc[X, "erro_abs_soma"].sum() / g["erro_abs_soma"].sum()),
                     "MAE_agregado_sem_erro_desses": mae2, "reducao_MAE_agregado_%": 100 * (1 - mae2 / mae)}
            if mase_ref and m in mase_ref:
                linha["MASE_agregado"] = mase_ref[m]
                linha["MASE_agregado_sem_erro_desses"] = mase_ref[m] * mae2 / mae
            resumo.append(linha)
        pior = g.sort_values("erro_abs_soma", ascending=False).head(3).reset_index()
        pior.insert(0, "modelo", m)
        casos.append(pior)
    return pd.DataFrame(resumo), pd.concat(casos, ignore_index=True)


# ------------------------------------------------------------------ 7. vendedores sem venda no teste
def vendedores_inativos_teste(serie, perfis, comparison, hist, train_target_end, modelos, cortes=("2018-04-30",)):
    """Quantos vendedores da amostra não venderam no teste e como as métricas ficam só com os ativos."""
    from .metricas import summarize_sellers
    per = perfis.set_index("seller_id")["perfil"].astype(str)
    te = serie.loc[serie["week_start"] > train_target_end]
    s = te.groupby("seller_id").agg(receita=("revenue", "sum"))
    s["ultima_venda"] = te.loc[te["revenue"] > 0].groupby("seller_id")["week_start"].max()
    s["perfil"] = s.index.map(per)
    inat = s.index[s["receita"] == 0]
    contagem = [{"grupo": "sem venda nas 34 semanas do teste", "vendedores": len(inat),
                 **{p: int((s.loc[inat, "perfil"] == p).sum()) for p in ["intermitente", "medio", "regular"]}}]
    for c in cortes:
        ids = s.index[(s["receita"] > 0) & (s["ultima_venda"] < pd.Timestamp(c))]
        contagem.append({"grupo": f"venderam no teste, mas não depois de {pd.Timestamp(c):%d/%m/%Y}",
                         "vendedores": len(ids), **{p: int((s.loc[ids, "perfil"] == p).sum())
                                                    for p in ["intermitente", "medio", "regular"]}})
    total = s["perfil"].value_counts()
    contagem.append({"grupo": "amostra", "vendedores": len(s), **{p: int(total.get(p, 0))
                                                                 for p in ["intermitente", "medio", "regular"]}})
    scale = seller_scales(hist, train_target_end)
    y = TARGET
    linhas = []
    for nome, df in [("todos", comparison), ("só ativos no teste", comparison.loc[~comparison["seller_id"].isin(inat)])]:
        _, per_v = evaluate_seller_level(df, modelos, scale)
        resumo = summarize_sellers(per_v).set_index("modelo")
        agg = df.groupby("target_week")[[y] + [f"pred_{m}" for m in modelos]].sum()
        mae_naive = (df[y] - df["pred_Naive"]).abs().mean()
        for m in modelos:
            mae = (df[y] - df[f"pred_{m}"]).abs().mean()
            linhas.append({"conjunto": nome, "vendedores": df["seller_id"].nunique(), "modelo": m,
                           "MASE_medio_vendedor": resumo.loc[m, "MASE_media"],
                           "ganho_MAE_vs_Naive_%": 100 * (1 - mae / mae_naive),
                           "%_vendedores_com_ganho": 100 * resumo.loc[m, "prop_vendedores_melhora"],
                           "WAPE_agregado": (agg[y] - agg[f"pred_{m}"]).abs().sum() / agg[y].sum(),
                           "vies_total_%": 100 * (agg[f"pred_{m}"].sum() / agg[y].sum() - 1)})
    por_perfil = []
    escolhido = modelos[1]
    for nome, df in [("todos", comparison), ("só ativos no teste", comparison.loc[~comparison["seller_id"].isin(inat)])]:
        for p in ["intermitente", "medio", "regular"]:
            d = df.loc[df["seller_id"].map(per) == p]
            _, pv = evaluate_seller_level(d, ["Naive", escolhido], scale)
            r = summarize_sellers(pv).set_index("modelo")
            mae = (d[y] - d[f"pred_{escolhido}"]).abs().mean()
            mae_n = (d[y] - d["pred_Naive"]).abs().mean()
            por_perfil.append({"conjunto": nome, "perfil": p, "vendedores": d["seller_id"].nunique(),
                               f"MASE_{escolhido}": r.loc[escolhido, "MASE_media"], "MASE_Naive": r.loc["Naive", "MASE_media"],
                               "ganho_MAE_vs_Naive_%": 100 * (1 - mae / mae_n),
                               "%_vendedores_com_ganho": 100 * r.loc[escolhido, "prop_vendedores_melhora"]})
    return pd.DataFrame(contagem), pd.DataFrame(linhas), pd.DataFrame(por_perfil)


# ------------------------------------------------------------------ 8. faixas das variáveis macro
def faixas_macro(exog_semanal, macro_cols, panel_start, train_target_end, test_end):
    """Mínimo e máximo de cada variável macro (já com a defasagem de 2 meses) nas semanas-alvo do treino e do
    teste, e a proporção de semanas do teste fora da faixa do treino."""
    e = exog_semanal.set_index("target_week")
    tr = e.loc[(e.index >= panel_start) & (e.index <= train_target_end)]
    te = e.loc[(e.index > train_target_end) & (e.index <= test_end)]
    rows = []
    for c in macro_cols:
        lo, hi = tr[c].min(), tr[c].max()
        rows.append({"variavel": c, "treino_min": lo, "treino_max": hi, "teste_min": te[c].min(), "teste_max": te[c].max(),
                     "%_semanas_teste_abaixo": 100 * float((te[c] < lo).mean()),
                     "%_semanas_teste_acima": 100 * float((te[c] > hi).mean()),
                     "%_semanas_teste_fora": 100 * float(((te[c] < lo) | (te[c] > hi)).mean()),
                     "meses_referencia_treino": f"{pd.Timestamp(tr['macro_reference_month'].min()):%m/%Y} a {pd.Timestamp(tr['macro_reference_month'].max()):%m/%Y}",
                     "meses_referencia_teste": f"{pd.Timestamp(te['macro_reference_month'].min()):%m/%Y} a {pd.Timestamp(te['macro_reference_month'].max()):%m/%Y}"})
    return pd.DataFrame(rows)
