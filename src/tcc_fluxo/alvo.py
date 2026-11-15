"""Seções 1–5 do v5: leitura, construção do alvo, auditorias e corte TEST_END."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_items_dataset.csv",
]


def ler_olist(data_dir: Path):
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes em {data_dir.resolve()}: {missing}. Execução interrompida.")
    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp", "order_approved_at"])
    payments = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    return orders, payments, items


def diagnostico_bruto(orders, payments, items):
    miss_appr = orders["order_approved_at"].isna()
    pay_zero_na = payments["payment_value"].isna() | payments["payment_value"].le(0)
    pay_order_total = payments.groupby("order_id")["payment_value"].sum()
    diag = [
        ("Linhas em orders", f"{len(orders):,}"),
        ("Linhas em payments", f"{len(payments):,}"),
        ("Linhas em items", f"{len(items):,}"),
        ("Pedidos distintos (orders)", f"{orders['order_id'].nunique():,}"),
        ("Pedidos com pagamento registrado", f"{payments['order_id'].nunique():,}"),
        ("Pedidos com itens", f"{items['order_id'].nunique():,}"),
        ("Vendedores com itens", f"{items['seller_id'].nunique():,}"),
        ("order_approved_at ausente", f"{miss_appr.sum():,} ({miss_appr.mean():.3%})"),
        ("Registros de pagamento com valor 0 ou ausente", f"{pay_zero_na.sum():,} ({pay_zero_na.mean():.4%})"),
        ("Pedidos com soma de pagamentos = 0", f"{int(pay_order_total.le(0).sum()):,}"),
        ("Intervalo de compra", f"{orders['order_purchase_timestamp'].min()} a {orders['order_purchase_timestamp'].max()}"),
        ("Intervalo de aprovação", f"{orders['order_approved_at'].min()} a {orders['order_approved_at'].max()}"),
    ]
    tab = pd.DataFrame(diag, columns=["Diagnóstico", "Valor"]).set_index("Diagnóstico")
    status = orders.loc[miss_appr, "order_status"].value_counts().rename("pedidos").to_frame()
    return tab, status


def escolher_coluna_data(orders, max_share):
    """Regra fixada antes dos modelos: <= max_share ausente -> exclui; senão usa data da compra."""
    miss = orders["order_approved_at"].isna()
    share = float(miss.mean())
    if share <= max_share:
        return {"DATE_STRATEGY": "approved", "DATE_COL": "order_approved_at",
                "TARGET_NAME": "proxy de entrada bruta de caixa (pagamentos aprovados, alocados por vendedor)",
                "share_missing": share, "n_missing": int(miss.sum())}
    return {"DATE_STRATEGY": "purchase", "DATE_COL": "order_purchase_timestamp",
            "TARGET_NAME": "receita bruta de pedidos por semana",
            "share_missing": share, "n_missing": int(miss.sum())}


def receita_pedido_vendedor(orders, payments, items, date_strategy):
    """Seção 2: aloca o pagamento total do pedido entre vendedores pela participação no valor bruto."""
    payment_by_order = payments.groupby("order_id", as_index=False).agg(payment_total=("payment_value", "sum"))
    item_by_seller_order = (
        items.assign(item_gross=items["price"] + items["freight_value"])
        .groupby(["order_id", "seller_id"], as_index=False)
        .agg(item_gross=("item_gross", "sum"))
    )
    item_by_seller_order["order_gross"] = item_by_seller_order.groupby("order_id")["item_gross"].transform("sum")
    item_by_seller_order["seller_share"] = np.where(
        item_by_seller_order["order_gross"] > 0,
        item_by_seller_order["item_gross"] / item_by_seller_order["order_gross"],
        0.0,
    )
    osr = item_by_seller_order.merge(payment_by_order, on="order_id", how="left", validate="many_to_one")
    osr["allocated_revenue"] = osr["payment_total"].fillna(0) * osr["seller_share"]
    osr = osr.merge(orders[["order_id", "order_approved_at", "order_purchase_timestamp", "order_status"]],
                    on="order_id", how="left", validate="many_to_one")

    n_before = len(osr)
    keep = osr["payment_total"].fillna(0).gt(0)
    if date_strategy == "approved":
        keep &= osr["order_approved_at"].notna()
    excluded = osr.loc[~keep]
    osr = osr.loc[keep].copy()
    resumo = {
        "linhas_antes": n_before,
        "excluidas_linhas": len(excluded), "excluidos_pedidos": int(excluded["order_id"].nunique()),
        "excluidas_sem_aprovacao": int(excluded["order_approved_at"].isna().sum()),
        "excluidas_sem_pagamento": int(excluded["payment_total"].fillna(0).le(0).sum()),
        "mantidas_linhas": len(osr), "mantidos_pedidos": int(osr["order_id"].nunique()),
    }
    return payment_by_order, item_by_seller_order, osr, resumo


def agregar_semanal(order_seller_revenue, date_col):
    """Seção 3: semana seg–dom pela data do alvo. Acrescenta week_start em order_seller_revenue."""
    ts = pd.to_datetime(order_seller_revenue[date_col])
    order_seller_revenue["week_start"] = ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")
    weekly = (order_seller_revenue.groupby(["seller_id", "week_start"], as_index=False)
              .agg(cash_in_proxy=("allocated_revenue", "sum"), orders=("order_id", "nunique")))
    assert (weekly["week_start"].dt.weekday == 0).all()
    return weekly


def auditorias(orders, payments, items, payment_by_order, item_by_seller_order,
               order_seller_revenue, weekly_observed, date_strategy):
    """Seção 4: unicidade, não negatividade e reconciliação. Para a execução se algo falhar."""
    assert payment_by_order["order_id"].is_unique
    assert orders["order_id"].is_unique
    assert not item_by_seller_order.duplicated(["order_id", "seller_id"]).any()
    assert not weekly_observed.duplicated(["seller_id", "week_start"]).any()
    assert (weekly_observed["cash_in_proxy"] >= 0).all()

    if date_strategy == "approved":
        approved_orders = orders.loc[orders["order_approved_at"].notna(), ["order_id"]]
    else:
        approved_orders = orders[["order_id"]]
    approved_pay = approved_orders.merge(payment_by_order, on="order_id", how="inner")
    approved_pay = approved_pay.loc[approved_pay["payment_total"].gt(0)]
    total_approved = approved_pay["payment_total"].sum()
    allocated_total = order_seller_revenue["allocated_revenue"].sum()
    diff_abs = total_approved - allocated_total

    ids_with_items = set(item_by_seller_order["order_id"])
    no_items = approved_pay.loc[~approved_pay["order_id"].isin(ids_with_items)]
    zero_gross_ids = set(item_by_seller_order.loc[item_by_seller_order["order_gross"].le(0), "order_id"])
    zero_gross = approved_pay.loc[approved_pay["order_id"].isin(zero_gross_ids)]
    alloc_by_seller = order_seller_revenue.groupby("seller_id")["allocated_revenue"].sum()
    sellers_no_revenue = set(items["seller_id"]) - set(alloc_by_seller.index[alloc_by_seller.gt(0)])

    recon = pd.DataFrame([
        ("Soma dos pagamentos aprovados (R$)", total_approved),
        ("Soma alocada aos vendedores (R$)", allocated_total),
        ("Diferença absoluta (R$)", diff_abs),
        ("Diferença percentual (%)", 100 * diff_abs / total_approved),
        ("Pedidos aprovados com pagamento e SEM itens", len(no_items)),
        ("  valor desses pedidos (R$)", no_items["payment_total"].sum()),
        ("Pedidos com valor bruto de itens = 0", len(zero_gross)),
        ("Vendedores com itens mas sem receita alocada", len(sellers_no_revenue)),
    ], columns=["Reconciliação", "Valor"]).set_index("Reconciliação")
    explained = no_items["payment_total"].sum() + zero_gross["payment_total"].sum()
    assert np.isclose(diff_abs, explained, atol=0.05), (diff_abs, explained)
    return recon


def composicao_status(order_seller_revenue):
    mix = (order_seller_revenue.groupby("order_status")
           .agg(pedidos=("order_id", "nunique"), valor=("allocated_revenue", "sum"))
           .sort_values("valor", ascending=False))
    mix["participacao_valor_%"] = 100 * mix["valor"] / mix["valor"].sum()
    return mix


def corte_test_end(order_seller_revenue, weekly_observed, date_col, tail_gap_days, tail_max_orders,
                   train_target_end):
    """Seção 5: TEST_END = segunda-feira da última semana completa, com resíduo de cauda removido."""
    order_level = order_seller_revenue.drop_duplicates("order_id")[["order_id", date_col, "week_start"]]
    daily = order_level[date_col].dt.normalize().value_counts().sort_index()
    gaps = pd.Series(daily.index, index=daily.index).diff().dt.days

    last_regular_day = daily.index.max()
    residual_days = []
    big_gaps = gaps[gaps >= tail_gap_days]
    if len(big_gaps):
        first_after_gap = big_gaps.index[-1]
        tail = daily.loc[first_after_gap:]
        if tail.sum() <= tail_max_orders:
            residual_days = list(tail.index)
            last_regular_day = daily.loc[:first_after_gap].index[-2]

    weeks_info = order_level.groupby("week_start").agg(pedidos=("order_id", "nunique"),
                                                       ultimo_registro=(date_col, "max"))
    weeks_info["valor"] = weekly_observed.groupby("week_start")["cash_in_proxy"].sum()
    weeks_info["domingo"] = weeks_info.index + pd.Timedelta(days=6)
    weeks_info["completa"] = weeks_info["domingo"] <= last_regular_day

    test_end = weeks_info.index[weeks_info["completa"]].max()
    removed = list(weeks_info.index[weeks_info.index > test_end])
    assert test_end + pd.Timedelta(days=6) <= last_regular_day
    assert test_end > train_target_end
    return {
        "TEST_END": test_end, "last_regular_day": last_regular_day, "residual_days": residual_days,
        "residual_orders": int(daily.loc[residual_days].sum()) if residual_days else 0,
        "removed_weeks": removed, "weeks_info": weeks_info, "daily": daily,
    }


def comparar_status(osr, cohort_ids, panel_start, test_end, train_target_end):
    """Compara duas definições do alvo na coorte: todos os pedidos aprovados (atual) × só 'delivered'.

    Usa o status FINAL de cada pedido na base (fotografia da extração). Descritivo: não ajusta modelos.
    """
    o = osr.loc[osr["seller_id"].isin(cohort_ids) & osr["week_start"].between(panel_start, test_end)].copy()
    o["entregue"] = o["order_status"].eq("delivered")
    o["periodo"] = np.where(o["week_start"] <= train_target_end, "treino (2017)", "teste (2018)")

    por_status = (o.groupby("order_status")
                  .agg(pedidos=("order_id", "nunique"), valor=("allocated_revenue", "sum"))
                  .sort_values("valor", ascending=False))
    por_status["pct_valor"] = 100 * por_status["valor"] / por_status["valor"].sum()

    def _perda(g):
        tot = g["allocated_revenue"].sum()
        fora = g.loc[~g["entregue"], "allocated_revenue"].sum()
        return pd.Series({"valor_todos": tot, "valor_so_entregues": tot - fora, "pct_removido": 100 * fora / tot})

    por_periodo = o.groupby("periodo")[["allocated_revenue", "entregue"]].apply(_perda)
    por_semana = o.groupby("week_start")[["allocated_revenue", "entregue"]].apply(_perda).reset_index()

    a = o.groupby(["seller_id", "week_start"])["allocated_revenue"].sum()
    b = (o.loc[o["entregue"]].groupby(["seller_id", "week_start"])["allocated_revenue"].sum()
         .reindex(a.index, fill_value=0.0))
    por_vendedor = pd.DataFrame({"valor_todos": a.groupby(level=0).sum(), "valor_so_entregues": b.groupby(level=0).sum()})
    por_vendedor["pct_removido"] = 100 * (1 - por_vendedor["valor_so_entregues"] / por_vendedor["valor_todos"])
    por_vendedor = por_vendedor.sort_values("pct_removido", ascending=False).reset_index()

    tot_a, tot_b = a.groupby(level=1).sum(), b.groupby(level=1).sum()
    resumo = {
        "pct_valor_removido_total": float(100 * (1 - b.sum() / a.sum())),
        "pct_valor_cancelado": float(por_status["pct_valor"].get("canceled", 0.0)),
        "pct_vendedor_semanas_com_valor_que_mudam": float(100 * (~np.isclose(a.to_numpy(), b.to_numpy())).mean()),
        "vendedor_semanas_que_viram_zero": int(((a > 0) & (b == 0)).sum()),
        "vendedores_com_mais_de_5pct_removido": int((por_vendedor["pct_removido"] > 5).sum()),
        "vendedores_com_mais_de_20pct_removido": int((por_vendedor["pct_removido"] > 20).sum()),
        "correlacao_total_semanal": float(np.corrcoef(tot_a, tot_b)[0, 1]),
        "maior_pct_removido_numa_semana": float(por_semana["pct_removido"].max()),
    }
    return {"resumo": resumo, "por_status": por_status.reset_index(), "por_periodo": por_periodo.reset_index(),
            "por_semana": por_semana, "por_vendedor": por_vendedor}
