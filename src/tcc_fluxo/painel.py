"""Seções 6–11 do v5: coorte, painel vendedor-semana, atributos sem vazamento e divisão temporal."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "target_next_week"
CAT_FEATURES = ["seller_id"]
BASE_NUMERIC_FEATURES = [
    "revenue_current", "orders_current",
    "rev_lag_1", "rev_lag_2", "rev_lag_3", "rev_lag_4", "rev_lag_8",
    "rev_mean_4", "rev_mean_8", "rev_std_4", "rev_expanding_mean",
    "zero_weeks_prev_4", "weeks_since_start",
    "target_month", "target_quarter", "target_week_sin", "target_week_cos",
]


@dataclass
class Especificacao:
    """Listas de atributos e parâmetros de ajuste compartilhados por todos os modelos."""
    numeric: list
    exog: list
    seed: int = 42
    n_threads: int = 1
    cat: list = field(default_factory=lambda: list(CAT_FEATURES))
    target: str = TARGET

    @property
    def numeric_exog(self):
        return self.numeric + self.exog

    @classmethod
    def de_json(cls, spec: dict, C):
        return cls(numeric=list(spec["BASE_NUMERIC_FEATURES"]), exog=list(spec["EXOGENOUS_FEATURES"]),
                   seed=C.RANDOM_SEED, n_threads=C.N_THREADS)


def coorte(weekly_observed, panel_start, train_end, min_calendar_weeks, min_sales_weeks,
           use_mei_proxy=False, mei_limit=None):
    """Seção 6: elegibilidade só com informação do treino."""
    observed_train = weekly_observed.loc[weekly_observed["week_start"].between(panel_start, train_end)]
    elig = observed_train.groupby("seller_id").agg(
        first_train_week=("week_start", "min"), last_train_week=("week_start", "max"),
        sales_weeks=("week_start", "nunique"), revenue_train=("cash_in_proxy", "sum"))
    elig["calendar_weeks"] = (elig["last_train_week"] - elig["first_train_week"]).dt.days // 7 + 1
    elig["ok_calendar"] = elig["calendar_weeks"] >= min_calendar_weeks
    elig["ok_sales"] = elig["sales_weeks"] >= min_sales_weeks
    months_active = ((train_end - elig["first_train_week"]).dt.days + 1) / 30.4375
    elig["avg_monthly_revenue_train"] = elig["revenue_train"] / months_active
    weeks_active = (train_end - elig["first_train_week"]).dt.days // 7 + 1
    elig["avg_weekly_revenue_train"] = elig["revenue_train"] / weeks_active
    elig["eligible"] = elig["ok_calendar"] & elig["ok_sales"]
    if use_mei_proxy:
        elig["ok_mei_proxy"] = elig["avg_monthly_revenue_train"] <= mei_limit
        elig["eligible"] &= elig["ok_mei_proxy"]
        label = (f"vendedores com receita média mensal no treino <= R$ {mei_limit:,.2f} "
                 "(proxy de porte; NÃO comprova enquadramento como MEI)")
    else:
        label = "vendedores elegíveis segundo o critério semanal definido"
    cohort = elig.loc[elig["eligible"]].copy()
    return elig, cohort, label, observed_train


def montar_panel_base(cohort, weekly_observed, test_end, panel_start):
    """Seção 7: grade semanal da primeira semana do vendedor até TEST_END, zeros preenchidos."""
    first_week = cohort["first_train_week"]
    pb = pd.concat([pd.DataFrame({"seller_id": sid, "week_start": pd.date_range(fw, test_end, freq="7D")})
                    for sid, fw in first_week.items()], ignore_index=True)
    pb = pb.merge(weekly_observed.loc[weekly_observed["seller_id"].isin(cohort.index)],
                  on=["seller_id", "week_start"], how="left", validate="one_to_one")
    pb["had_record"] = pb["cash_in_proxy"].notna().astype("int8")
    pb["revenue"] = pb["cash_in_proxy"].fillna(0.0)
    pb["orders"] = pb["orders"].fillna(0).astype(int)
    pb = pb.drop(columns="cash_in_proxy").sort_values(["seller_id", "week_start"]).reset_index(drop=True)

    obs = weekly_observed.loc[weekly_observed["seller_id"].isin(cohort.index)
                              & weekly_observed["week_start"].ge(panel_start)]
    assert np.isclose(obs["cash_in_proxy"].sum(), pb["revenue"].sum())
    assert (pb.groupby("seller_id")["week_start"].diff().dropna() == pd.Timedelta(days=7)).all()
    return pb


def build_features(df):
    """Seções 8–10. A linha da semana t prevê t+1; tudo que é atributo usa só até t."""
    df = df.sort_values(["seller_id", "week_start"]).reset_index(drop=True).copy()
    g = df.groupby("seller_id", sort=False)
    df["revenue_current"] = df["revenue"]
    df["orders_current"] = df["orders"]
    for lag in (1, 2, 3, 4, 8):
        df[f"rev_lag_{lag}"] = g["revenue"].shift(lag)
    df["rev_mean_4"] = g["revenue"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["rev_mean_8"] = g["revenue"].transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    df["rev_std_4"] = g["revenue"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
    df["rev_expanding_mean"] = g["revenue"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["zero_weeks_prev_4"] = g["revenue"].transform(
        lambda s: s.shift(1).eq(0).astype(float).where(s.shift(1).notna()).rolling(4, min_periods=1).sum())
    df["weeks_since_start"] = g.cumcount()
    df["target_week"] = g["week_start"].shift(-1)
    df["target_week_of_year"] = df["target_week"].dt.isocalendar().week.astype("float64")
    df["target_month"] = df["target_week"].dt.month.astype("float64")
    df["target_quarter"] = df["target_week"].dt.quarter.astype("float64")
    df["target_week_sin"] = np.sin(2 * np.pi * df["target_week_of_year"] / 52.0)
    df["target_week_cos"] = np.cos(2 * np.pi * df["target_week_of_year"] / 52.0)
    df[TARGET] = g["revenue"].shift(-1)
    return df


def attach_exogenous_features(df, exog_weekly):
    out = build_features(df)
    return out.merge(exog_weekly, on="target_week", how="left", validate="many_to_one")


def teste_vazamento(panel_base, panel, exog_weekly, feature_cols, cut, seed):
    """Perturbar TODO o futuro depois de `cut` não pode mudar nenhum atributo até `cut`."""
    rng = np.random.default_rng(seed)
    perturbed = panel_base.copy()
    fut = perturbed["week_start"] > cut
    perturbed.loc[fut, "revenue"] = perturbed.loc[fut, "revenue"] * rng.uniform(0.5, 3.0, fut.sum()) + 1000.0
    perturbed.loc[fut, "orders"] = perturbed.loc[fut, "orders"] + 7
    pf = attach_exogenous_features(perturbed, exog_weekly)
    past = panel["week_start"] <= cut
    pd.testing.assert_frame_equal(panel.loc[past, feature_cols], pf.loc[past, feature_cols])
    changed = (panel.loc[past & panel["target_week"].gt(cut), TARGET]
               != pf.loc[past & pf["target_week"].gt(cut), TARGET]).mean()
    return perturbed, float(changed)


def dividir(panel, all_numeric_features, train_target_end, test_end):
    """Seção 11: divisão pela data do ALVO. Devolve o painel com a coluna `conjunto`."""
    usable_mask = panel[[TARGET] + all_numeric_features].notna().all(axis=1)
    tw = panel["target_week"]
    is_train = usable_mask & (tw <= train_target_end)
    is_test = usable_mask & (tw > train_target_end) & (tw <= test_end)
    out = panel.copy()
    out["conjunto"] = np.select([is_train, is_test], ["treino", "teste"], default="fora")

    train, test = out.loc[is_train], out.loc[is_test]
    assert set(train["target_week"]).isdisjoint(set(test["target_week"]))
    assert train["target_week"].max() <= train_target_end
    assert test["target_week"].max() == test_end
    n_possible = panel["target_week"].between(train_target_end + pd.Timedelta(days=1), test_end).sum()
    assert len(test) == n_possible, (len(test), n_possible)
    resumo = {
        "linhas_com_alvo": int(panel[TARGET].notna().sum()), "linhas_utilizaveis": int(usable_mask.sum()),
        "treino_linhas": len(train), "treino_vendedores": int(train["seller_id"].nunique()),
        "treino_alvos": [str(train["target_week"].min().date()), str(train["target_week"].max().date())],
        "treino_semanas": int(train["target_week"].nunique()),
        "teste_linhas": len(test), "teste_vendedores": int(test["seller_id"].nunique()),
        "teste_alvos": [str(test["target_week"].min().date()), str(test["target_week"].max().date())],
        "teste_semanas": int(test["target_week"].nunique()),
        "alvos_zero_treino": float(train[TARGET].eq(0).mean()),
        "alvos_zero_teste": float(test[TARGET].eq(0).mean()),
    }
    return out, resumo


def perfis_treino(cohort):
    """Seção 20: tercis da frequência de semanas com venda no treino."""
    profile = cohort[["sales_weeks", "calendar_weeks"]].copy()
    profile["freq_venda_treino"] = profile["sales_weeks"] / profile["calendar_weeks"]
    profile["perfil"] = pd.qcut(profile["freq_venda_treino"], 3, labels=["intermitente", "medio", "regular"])
    return profile


def resumo_porte(cohort, limites: dict):
    """Quantos vendedores da coorte passam de cada limite mensal (descritivo; não filtra ninguém).

    Usa `avg_monthly_revenue_train`: receita do treino (2017) dividida pelos meses entre a primeira
    semana com venda e TRAIN_END, ou seja, média proporcional ao período de atividade. A receita é a
    proxy do alvo (pagamentos aprovados alocados, com frete), não o faturamento fiscal do vendedor.
    """
    m = cohort["avg_monthly_revenue_train"]
    rows = [{"limite": nome, "limite_mensal_R$": float(v), "vendedores_acima": int((m > v).sum()),
             "pct_coorte": float((m > v).mean()), "n_coorte": int(len(m))}
            for nome, v in limites.items()]
    return pd.DataFrame(rows)
