"""Notebook 12 (OE3): padrões sazonais da receita em 2017 e associação com o calendário comercial.

Descritivo e exploratório. Usa só semanas até TRAIN_TARGET_END (treino); não toca no teste.
Cada evento ocorre uma única vez em 2017: os números são associações observadas, não efeitos causais.

Lift de uma semana = receita da semana ÷ média das 4 semanas anteriores à JANELA do evento, pulando semanas de
outros eventos comerciais e a semana seguinte à Black Friday (efeito residual), para que um evento não contamine
a base de outro (ex.: a Black Friday cairia na base do Natal).
Como a coorte cresce ao longo de 2017, um lift > 1 também reflete tendência: por isso cada lift é
comparado com a mediana dos lifts das semanas comuns (sem evento comercial e sem feriado).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (rótulo, chave no calendário do projeto, semanas antes do evento que entram na janela)
EVENTOS_PADRAO = [
    ("Black Friday", "commercial_black_friday", 0),
    ("Natal e Réveillon", "holiday_christmas", 1),     # 2017: 25/12 (seg) e 31/12 (dom) na mesma semana; a anterior é a de compras
    ("Dia das Mães", "commercial_mothers_day", 1),
    ("Dia dos Namorados", "commercial_valentines_day", 1),
    ("Dia dos Pais", "commercial_fathers_day", 1),
    ("Dia do Consumidor", "commercial_consumer_day", 0),
]
BASE_SEMANAS = 4


def segunda(d):
    d = pd.Timestamp(d)
    return d - pd.Timedelta(days=d.weekday())


def receita_semanal(hist, inicio, fim):
    tot = hist.loc[hist["week_start"].between(inicio, fim)].groupby("week_start")["revenue"].sum()
    ativos = (hist.loc[hist["week_start"].between(inicio, fim)].groupby("week_start")["seller_id"].nunique())
    return tot, ativos


def base_limpa(tot, primeira_semana, excluir, max_recuo=12):
    """Média das 4 semanas mais recentes antes de `primeira_semana` que não estão em `excluir`."""
    semanas = []
    for k in range(1, max_recuo + 1):
        s = primeira_semana - pd.Timedelta(weeks=k)
        if s not in tot.index:
            break
        if s not in excluir:
            semanas.append(s)
        if len(semanas) == BASE_SEMANAS:
            return float(tot.loc[semanas].mean()), sorted(semanas)
    return np.nan, sorted(semanas)


def semanas_excluidas(cal, ano, eventos):
    """Semanas de janelas de eventos comerciais + semana seguinte à Black Friday."""
    jan = janelas_eventos(cal, ano, eventos)
    excl = set(jan["semana"])
    bf = jan.loc[jan["chave"] == "commercial_black_friday", "semana"]
    excl |= {w + pd.Timedelta(weeks=1) for w in bf}
    return excl


def lifts_semanais(tot, excluir):
    """Lift de cada semana contra a base limpa (para a referência de semanas comuns e para os feriados)."""
    vals = {s: base_limpa(tot, s, excluir)[0] for s in tot.index}
    base = pd.Series(vals)
    return (tot / base).dropna()


def _fmt_semanas(ss):
    return ", ".join(f"{s:%d/%m}" for s in ss)


def semanas_feriado(cal, ano):
    f = cal.loc[(cal["ano"] == ano) & cal["evento"].str.startswith("holiday_")].copy()
    f["semana"] = f["date"].map(segunda)
    return f


def janelas_eventos(cal, ano, eventos=EVENTOS_PADRAO):
    linhas = []
    for rotulo, chave, antes in eventos:
        datas = cal.loc[(cal["ano"] == ano) & (cal["evento"] == chave), "date"]
        if datas.empty:
            continue
        data = pd.Timestamp(datas.min())
        w = segunda(data)
        for k in range(antes, -1, -1):
            linhas.append({"evento": rotulo, "chave": chave, "data": data.date(), "semana": w - pd.Timedelta(weeks=k),
                           "janela": "semana do evento" if k == 0 else f"{k} semana(s) antes",
                           "primeira_semana_janela": w - pd.Timedelta(weeks=antes)})
    return pd.DataFrame(linhas)


def tabela_lift(tot, cal, ano, eventos=EVENTOS_PADRAO):
    jan = janelas_eventos(cal, ano, eventos)
    fer = semanas_feriado(cal, ano)
    excl = semanas_excluidas(cal, ano, eventos)
    comuns = lifts_semanais(tot, excl)
    comuns = comuns.loc[~comuns.index.isin(excl | set(fer["semana"]))]
    ref = float(comuns.median())
    rows = []
    for _, r in jan.iterrows():
        base, semanas_base = base_limpa(tot, r["primeira_semana_janela"], excl)
        receita = float(tot.get(r["semana"], np.nan))
        sobrepoe = fer.loc[(fer["semana"] == r["semana"]), "evento"].tolist()
        rows.append({"evento": r["evento"], "data_evento": r["data"], "janela": r["janela"],
                     "semana": f"{r['semana']:%d/%m} a {r['semana'] + pd.Timedelta(days=6):%d/%m/%Y}",
                     "receita_semana_R$": receita, "media_4_semanas_antes_R$": base,
                     "lift": receita / base if base > 0 else np.nan,
                     "lift_vs_semana_comum": (receita / base) / ref if base > 0 else np.nan,
                     "feriados_na_mesma_semana": ", ".join(x.removeprefix("holiday_") for x in sobrepoe),
                     "semanas_da_base": _fmt_semanas(semanas_base)})
    out = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    return out, {"lift_mediano_semanas_comuns": ref, "n_semanas_comuns": int(len(comuns))}


def tabela_feriados(tot, cal, ano, eventos=EVENTOS_PADRAO):
    """Lift de cada semana com feriado e comparação agrupada com as semanas comuns."""
    fer = semanas_feriado(cal, ano)
    excl = semanas_excluidas(cal, ano, eventos)
    lifts = lifts_semanais(tot, excl)
    jan = janelas_eventos(cal, ano, eventos)
    comerciais = set(jan["semana"])
    por = (fer.groupby("semana").agg(feriados=("evento", lambda s: ", ".join(x.removeprefix("holiday_") for x in s)),
                                     data=("date", "min")).reset_index())
    por["lift"] = por["semana"].map(lifts)
    por["tambem_evento_comercial"] = por["semana"].isin(comerciais)
    por = por.dropna(subset=["lift"])
    comuns = lifts.loc[~lifts.index.isin(set(fer["semana"]) | excl)]
    so_feriado = por.loc[~por["tambem_evento_comercial"], "lift"]
    resumo = {"semanas_com_feriado_avaliadas": int(len(por)),
              "semanas_so_feriado (sem evento comercial)": int(len(so_feriado)),
              "lift_medio_semanas_so_feriado": float(so_feriado.mean()),
              "lift_mediano_semanas_so_feriado": float(so_feriado.median()),
              "lift_medio_semanas_comuns": float(comuns.mean()),
              "lift_mediano_semanas_comuns": float(comuns.median()),
              "n_semanas_comuns": int(len(comuns))}
    por["semana"] = por["semana"].dt.date
    return por.sort_values("lift", ascending=False).reset_index(drop=True), resumo


def janela_black_friday(tot, data_bf, antes=2, depois=1, excluir=frozenset()):
    w = segunda(data_bf)
    primeira = w - pd.Timedelta(weeks=antes)
    base, _ = base_limpa(tot, primeira, set(excluir))
    rows = []
    for k in range(-antes, depois + 1):
        s = w + pd.Timedelta(weeks=k)
        if s not in tot.index:
            continue
        rows.append({"posicao": "semana da Black Friday" if k == 0 else (f"{-k} semana(s) antes" if k < 0 else f"{k} semana(s) depois"),
                     "semana": f"{s:%d/%m} a {s + pd.Timedelta(days=6):%d/%m}", "receita_R$": float(tot[s]),
                     "razao_vs_media_4_semanas_antes_da_janela": float(tot[s]) / base})
    return pd.DataFrame(rows), base


def black_friday_vendedores(hist, perfis, data_bf, semanas_ref=4):
    """Proporção de vendedores que venderam e que venderam acima da própria mediana (semanas com venda até a
    semana anterior), na semana da Black Friday e, como referência, na média das 4 semanas anteriores."""
    w = segunda(data_bf)
    h = hist.loc[hist["week_start"] <= w, ["seller_id", "week_start", "revenue"]]

    def medir(semana):
        antes = h.loc[(h["week_start"] < semana) & (h["revenue"] > 0)]
        mediana = antes.groupby("seller_id")["revenue"].median()
        s = h.loc[h["week_start"] == semana].set_index("seller_id")["revenue"]
        s = s.loc[s.index.isin(mediana.index)]
        return pd.DataFrame({"vendeu": s > 0, "acima_mediana": s > mediana.reindex(s.index)})

    bf = medir(w)
    refs = [medir(w - pd.Timedelta(weeks=k)) for k in range(1, semanas_ref + 1)]
    perfil = perfis.set_index("seller_id")["perfil"].astype(str)
    rows = []
    for grupo in ["todos", "intermitente", "medio", "regular"]:
        def prop(df, col):
            d = df if grupo == "todos" else df.loc[df.index.map(perfil) == grupo]
            return float(d[col].mean()), int(len(d))
        p_bf, n = prop(bf, "acima_mediana")
        v_bf, _ = prop(bf, "vendeu")
        p_ref = float(np.mean([prop(r, "acima_mediana")[0] for r in refs]))
        v_ref = float(np.mean([prop(r, "vendeu")[0] for r in refs]))
        rows.append({"perfil": grupo, "vendedores": n,
                     "% que venderam (Black Friday)": 100 * v_bf, "% que venderam (média 4 semanas antes)": 100 * v_ref,
                     "% acima da própria mediana (Black Friday)": 100 * p_bf,
                     "% acima da própria mediana (média 4 semanas antes)": 100 * p_ref})
    return pd.DataFrame(rows)


def importancia_grupos(prever, spec_json):
    """Participação de cada grupo de atributos na importância (ganho) do modelo, quando disponível."""
    modelo = getattr(prever, "modelo", None)
    attrs = getattr(prever, "atributos", None)
    if modelo is None:
        return None, None
    if hasattr(modelo, "booster_"):                       # LightGBM
        imp = pd.Series(modelo.booster_.feature_importance(importance_type="gain"), index=attrs)
    elif hasattr(modelo, "get_feature_importance"):       # CatBoost
        imp = pd.Series(modelo.get_feature_importance(), index=attrs)
    else:
        return None, None
    cal, macro = set(spec_json["CALENDAR_FEATURES"]), set(spec_json["MACRO_FEATURES"])
    grupo = {a: ("calendário (22)" if a in cal else "macroeconomia (4)" if a in macro
                 else "ID do vendedor" if a == "seller_id" else "histórico do vendedor e semana do ano (17)")
             for a in imp.index}
    share = 100 * imp / imp.sum()
    por_grupo = share.groupby(pd.Series(grupo)).sum().sort_values(ascending=False).rename("% da importância").reset_index()
    por_grupo.columns = ["grupo", "% da importância (ganho)"]
    detalhe = share.rename("% da importância (ganho)").reset_index().rename(columns={"index": "atributo"})
    detalhe["grupo"] = detalhe["atributo"].map(grupo)
    return por_grupo, detalhe.sort_values("% da importância (ganho)", ascending=False).reset_index(drop=True)
