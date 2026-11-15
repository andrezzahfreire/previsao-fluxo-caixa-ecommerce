"""Notebook 11: previsão recursiva da receita acumulada das próximas 4 semanas (análise complementar).

O resultado principal do TCC continua sendo h = 1. Aqui o modelo escolhido, treinado uma vez com todo o treino,
prevê t+1; essa previsão entra no lugar do realizado para montar os atributos de t+2, e assim até t+H.

Somar as previsões h = 1 do notebook 05 NÃO serve para isto: cada uma usa o realizado da semana anterior,
que ainda não existiria no dia da decisão.

Regras sem vazamento:
- receita das semanas ainda não realizadas = previsão (abaixo de R$ 1 vira 0, como nas semanas sem venda);
- pedidos dessas semanas = receita prevista ÷ ticket médio do vendedor até a origem;
- variáveis macro congeladas nos valores da semana t+1 (conhecidos na origem); calendário segue a semana-alvo,
  porque é conhecido de antemão.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .modelos import ajustar, check_and_clip
from .painel import TARGET, attach_exogenous_features


def origens(train_target_end, test_end, horizonte):
    """Origens O (fim da semana O conhecido) com as H semanas seguintes dentro do teste."""
    return pd.date_range(train_target_end, test_end - pd.Timedelta(weeks=horizonte), freq="7D")


def _ticket(hist):
    h = hist.loc[hist["orders"] > 0]
    t = (h.groupby("seller_id")["revenue"].sum() / h.groupby("seller_id")["orders"].sum())
    return t.replace([np.inf, -np.inf], np.nan)


def prever_origem(serie, exog, prever, spec, origem, horizonte, macro_cols, limiar_zero=1.0):
    """Previsões recursivas h = 1..H a partir de `origem`, para os vendedores com atributos completos em h = 1."""
    base = serie.loc[serie["week_start"] <= origem, ["seller_id", "week_start", "revenue", "orders"]].copy()
    vendedores = base.loc[base["week_start"] == origem, "seller_id"].unique()
    base = base.loc[base["seller_id"].isin(vendedores)]
    ticket = _ticket(base)
    feats_all = spec.numeric_exog
    saida, macro_h1, ativos = [], None, None
    frame = base
    for h in range(1, horizonte + 1):
        semana_t = origem + pd.Timedelta(weeks=h - 1)           # linha cujos atributos preveem a semana t+1
        alvo = semana_t + pd.Timedelta(weeks=1)
        marcador = pd.DataFrame({"seller_id": vendedores, "week_start": alvo, "revenue": np.nan, "orders": np.nan})
        feat = attach_exogenous_features(pd.concat([frame, marcador], ignore_index=True), exog)
        linhas = feat.loc[feat["week_start"] == semana_t].copy()
        if h == 1:
            ok = linhas[feats_all].notna().all(axis=1)
            linhas = linhas.loc[ok]
            ativos = linhas["seller_id"].to_numpy()
            macro_h1 = linhas.set_index("seller_id")[macro_cols]
        else:
            linhas = linhas.set_index("seller_id").loc[ativos].reset_index()
            linhas[macro_cols] = macro_h1.loc[ativos].to_numpy()
            assert linhas[feats_all].notna().all().all(), f"atributos ausentes em h={h}"
        pred, _ = check_and_clip(prever(linhas))
        saida.append(pd.DataFrame({"seller_id": linhas["seller_id"].to_numpy(), "origem": origem, "h": h,
                                   "semana_alvo": alvo, "previsto": pred}))
        rev_hat = np.where(pred >= limiar_zero, pred, 0.0)
        tk = pd.Series(linhas["seller_id"].to_numpy()).map(ticket).to_numpy(float)
        ord_hat = np.where(np.isfinite(tk) & (tk > 0), rev_hat / np.where(tk > 0, tk, 1.0), 0.0)
        novo = pd.DataFrame({"seller_id": linhas["seller_id"].to_numpy(), "week_start": alvo,
                             "revenue": rev_hat, "orders": ord_hat})
        frame = pd.concat([frame.loc[frame["seller_id"].isin(ativos)], novo], ignore_index=True)
    return pd.concat(saida, ignore_index=True)


def rodar(train, serie, exog, modelo, spec, C, test_end, horizonte=4, macro_cols=None, verbose=True):
    """Treina `modelo` uma vez com todo o treino e gera as previsões recursivas em todas as origens."""
    macro_cols = list(macro_cols or [])
    t0 = time.time()
    prever = ajustar(modelo, train, spec)
    partes = []
    orig = origens(C.TRAIN_TARGET_END, test_end, horizonte)
    for k, o in enumerate(orig, start=1):
        partes.append(prever_origem(serie, exog, prever, spec, o, horizonte, macro_cols))
        if verbose and (k % 5 == 0 or k == len(orig)):
            print(f"origem {o.date()} ({k}/{len(orig)}) | {time.time() - t0:5.0f}s")
    longo = pd.concat(partes, ignore_index=True)
    real = serie.set_index(["seller_id", "week_start"])["revenue"]
    longo["real"] = real.reindex(pd.MultiIndex.from_arrays([longo["seller_id"], longo["semana_alvo"]])).to_numpy()
    assert longo["real"].notna().all()
    return longo


def tabela_larga(longo, serie, horizonte=4):
    """Uma linha por vendedor × origem: previsões e realizados por semana, acumulados e referências ingênuas."""
    w = longo.pivot_table(index=["seller_id", "origem"], columns="h", values=["previsto", "real"])
    w.columns = [f"{a}_h{h}" for a, h in w.columns]
    w = w.reset_index()
    w["previsto_acum"] = w[[f"previsto_h{h}" for h in range(1, horizonte + 1)]].sum(axis=1)
    w["real_acum"] = w[[f"real_h{h}" for h in range(1, horizonte + 1)]].sum(axis=1)
    s = serie.sort_values(["seller_id", "week_start"]).copy()
    s["mm4"] = s.groupby("seller_id")["revenue"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    ref = s.set_index(["seller_id", "week_start"])
    chave = pd.MultiIndex.from_arrays([w["seller_id"], w["origem"]])
    w["naive_acum"] = horizonte * ref["revenue"].reindex(chave).to_numpy()
    w["mm4_acum"] = horizonte * ref["mm4"].reindex(chave).to_numpy()
    w["semana_inicio"] = w["origem"] + pd.Timedelta(weeks=1)
    w["semana_fim"] = w["origem"] + pd.Timedelta(weeks=horizonte)
    return w


def avaliar(larga, longo, modelo):
    """Erros do acumulado (vendedor × origem e total da coorte) e erro por horizonte."""
    y = larga["real_acum"].to_numpy(float)
    linhas = []
    for nome, col in [(modelo, "previsto_acum"), ("Naive × 4", "naive_acum"), ("MediaMovel4 × 4", "mm4_acum")]:
        p = larga[col].to_numpy(float)
        e = p - y
        tot = larga.groupby("origem")[["real_acum", col]].sum()
        linhas.append({
            "modelo": nome,
            "MAE_vendedor_R$": np.abs(e).mean(),
            "WAPE_vendedor": np.abs(e).sum() / np.abs(y).sum(),
            "vies_total_%": 100 * (p.sum() / y.sum() - 1),
            "%_vendedor_origens_subestimadas": 100 * (e < 0).mean(),
            "WAPE_total_coorte": (tot[col] - tot["real_acum"]).abs().sum() / tot["real_acum"].sum(),
        })
    res = pd.DataFrame(linhas)
    base = res.set_index("modelo").loc["Naive × 4", "MAE_vendedor_R$"]
    res["skill_vs_naive4"] = 1 - res["MAE_vendedor_R$"] / base

    em = np.abs(larga["previsto_acum"] - larga["real_acum"])
    en = np.abs(larga["naive_acum"] - larga["real_acum"])
    dif = ~np.isclose(em, en)
    freq = {"pct_mais_perto_que_naive4": float(100 * (em[dif] < en[dif]).mean()),
            "n_comparacoes_sem_empate": int(dif.sum()), "n_total": int(len(em))}

    por_h = (longo.assign(ae=lambda d: (d["previsto"] - d["real"]).abs(), e=lambda d: d["previsto"] - d["real"])
             .groupby("h").agg(MAE_R=("ae", "mean"), soma_real=("real", "sum"), soma_prev=("previsto", "sum"),
                               soma_ae=("ae", "sum")))
    por_h["WAPE"] = por_h["soma_ae"] / por_h["soma_real"]
    por_h["vies_%"] = 100 * (por_h["soma_prev"] / por_h["soma_real"] - 1)
    por_h = por_h.rename(columns={"MAE_R": "MAE_R$"})
    return res, freq, por_h[["MAE_R$", "WAPE", "vies_%"]].reset_index()
